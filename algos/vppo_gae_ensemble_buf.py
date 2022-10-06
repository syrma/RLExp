import os

import tensorflow as tf
import gym
import pybullet_envs
import time
import math
import argparse
import wandb
import sys
import tempfile
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Buffer(object):
    def __init__(self, obs_spc, act_spc, model, critics, size, gam=0.99, lam=0.97):
        self.ptr = 0
        self.last_idx = 0
        self.size = size
        self.continuous = bool(act_spc.shape)

        self.model = model
        self.critics = critics

        self.obs_buf = tf.TensorArray(obs_spc.dtype, size)
        self.act_buf = tf.TensorArray(act_spc.dtype, size)
        self.rew_buf = tf.TensorArray(tf.float32, size)
        self.prob_buf = tf.TensorArray(tf.float32, size)

        self.rets = []
        self.ret_rms = RunningMeanStd(shape=())
        self.lens = []

        self.V_hats = tf.TensorArray(tf.float32, size)
        self.gae = tf.TensorArray(tf.float32, size)

        self.gam = gam
        self.lam = lam

    # @tf.function
    def store(self, obs, act, rew, prob):
        self.obs_buf = self.obs_buf.write(self.ptr, obs)
        self.act_buf = self.act_buf.write(self.ptr, act)
        self.rew_buf = self.rew_buf.write(self.ptr, rew)
        self.prob_buf = self.prob_buf.write(self.ptr, prob)
        self.ptr += 1

    # @tf.function
    def finish_path(self, last_obs=None):
        current_episode = tf.range(self.last_idx, self.ptr)

        #bootstrapping the remaining values if the episode was interrupted
        if last_obs == None:
            last_val = 0
        else:
            predictions = [tf.squeeze(value_model((tf.expand_dims(last_obs, 0)))) for value_model in self.critics]
            last_val = tf.math.reduce_mean(predictions)

        # last_val = tf.squeeze(self.value_model(tf.expand_dims(last_obs, 0))) if last_obs is not None else 0

        length = self.ptr - self.last_idx
        ep_rew = self.rew_buf.gather(current_episode)
        ret = tf.reduce_sum(ep_rew) + last_val
        self.lens.append(length)
        self.rets.append(ret)

        #(attempt at) scaling the rewards
        self.ret_rms.update(np.array(self.rets))
        ep_rew = ep_rew / tf.sqrt(tf.cast(self.ret_rms.var, tf.float32) + 1e-8)

        # v_hats = discounted cumulative sum
        discounts = tf.math.cumprod(tf.fill(ep_rew.shape, self.gam), exclusive=True)
        v_hats = tf.math.cumsum(discounts * ep_rew, reverse=True)


        self.V_hats = self.V_hats.scatter(current_episode, v_hats)

        #Vs = tf.squeeze(value_model(self.obs_buf.gather(current_episode)), axis=1)

        predictions = [tf.squeeze(value_model(self.obs_buf.gather(current_episode)), axis=1) for value_model in self.critics]
        Vs = tf.math.reduce_mean(predictions, axis=0)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs

        # compute the advantage function (gae)
        discounts = tf.math.cumprod(tf.fill(deltas.shape, self.gam * self.lam), exclusive=True)
        gae = tf.math.cumsum(discounts * deltas, reverse=True)

        #Normalise the advantage
        gae = (gae - tf.math.reduce_mean(gae)) / (tf.math.reduce_std(gae) + 1e-8)

        self.gae = self.gae.scatter(current_episode, gae)

        self.last_idx = self.ptr

        if self.ptr == self.size:
            self.obs_buf = self.obs_buf.stack()
            self.act_buf = self.act_buf.stack()
            self.rew_buf = self.rew_buf.stack()
            self.prob_buf = self.prob_buf.stack()

            self.V_hats = self.V_hats.stack()
            self.gae = self.gae.stack()

    def approx_kl(self):
        obs, act, logprob = self.obs_buf, self.act_buf, self.prob_buf

        if self.continuous:
            dist = tfd.MultivariateNormalDiag(model(obs), tf.exp(self.model.log_std))
        else:
            dist = tfd.Categorical(logits=model(obs))

        new_logprob = dist.log_prob(act)

        return tf.reduce_mean(logprob - new_logprob)

    # @tf.function
    def loss(self):
        eps = 0.1
        obs, act, adv, logprob = self.obs_buf, self.act_buf, self.gae, self.prob_buf

        if self.continuous:
            dist = tfd.MultivariateNormalDiag(model(obs), tf.exp(self.model.log_std))
        else:
            dist = tfd.Categorical(logits=model(obs))

        new_logprob = dist.log_prob(act)

        mask = tf.cast(adv >= 0, tf.float32)
        epsilon_clip = mask * (1 + eps) + (1 - mask) * (1 - eps)
        ratio = tf.exp(new_logprob - logprob)

        return -tf.reduce_mean(tf.minimum(ratio * adv, epsilon_clip * adv))


@tf.function
def action(model, obs, env):
    est = tf.squeeze(model(tf.expand_dims(obs, 0)), axis=0)
    if env.action_space.shape:
        dist = tfd.MultivariateNormalDiag(est, tf.exp(model.log_std))
    else:
        dist = tfd.Categorical(logits=est, dtype=env.action_space.dtype)

    action = dist.sample()
    logprob = tf.reduce_sum(dist.log_prob(action))

    return action, logprob


def run_one_episode(env, buf):
    obs_dtype = env.observation_space.dtype

    obs = env.reset()
    obs = tf.cast(obs, obs_dtype)
    done = False

    for i in range(buf.ptr, buf.size):
        act, prob = action(buf.model, obs, env)
        new_obs, rew, done, _ = env.step(act.numpy())
        
        rew = tf.cast(rew, 'float32')

        buf.store(obs, act, rew, prob)
        obs = tf.cast(new_obs, obs_dtype)

        if done:
            break

    critic_start = time.time()
    if done:
        buf.finish_path()
    else:
        buf.finish_path(obs)

    return time.time() - critic_start


def train_one_epoch(env, batch_size, model, critics, γ, λ, save_dir):
    obs_spc = env.observation_space
    act_spc = env.action_space

    batch = Buffer(obs_spc, act_spc, model, critics, batch_size, gam=γ, lam=λ)
    start_time = time.time()

    critic_time = 0
    while batch.ptr < batch.size:
        critic_time += run_one_episode(env, batch)

    train_start_time = time.time()

    var_list = list(model.trainable_weights)
    if act_spc.shape:
        var_list.append(model.log_std)

    for i in range(80):
        save_model(model, save_dir)
        opt.minimize(batch.loss, var_list=var_list)

        # do we want early stopping?
        if not wandb.config.kl_stop:
            continue

        if batch.approx_kl() > 1.5 * kl_target:
            print(f"Early stopping at step {i}")
            # rollback if asked to
            if wandb.config.kl_rollback:
                load_model(model, save_dir)
            break

    train_time = time.time() - train_start_time
    run_time = train_start_time - start_time

    print('run time', run_time, 'critic time (included in run time):', critic_time, 'train time', train_time)
    print('AvgEpRet:', tf.reduce_mean(batch.rets).numpy())

    for i in range(len(critics)):
        bootstrap_value = 0.9 if wandb.config.bootstrap else 1
        mask = tf.random.uniform([batch.size]) < bootstrap_value
        masked_obs = tf.boolean_mask(batch.obs_buf, mask)
        masked_vhats = tf.boolean_mask(batch.V_hats, mask)
        hist = critics[i].fit(batch.obs_buf.numpy(), batch.V_hats.numpy(), epochs=80, steps_per_epoch=1, verbose=0)
        wandb.log({f'LossV{i}': tf.reduce_mean(hist.history['loss']).numpy()}, commit=False)

    wandb.log({'EpRet': wandb.Histogram(batch.rets),
               'AvgEpRet': tf.reduce_mean(batch.rets),
               'EpLen': tf.reduce_mean(batch.lens),
               'VVals': wandb.Histogram(batch.V_hats)},
              commit=False)

    return batch.rets, batch.lens

def save_model(model, save_path):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=None)
    manager.save()

def load_model(model, load_path):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint)
    print("Restoring from {}".format(manager.latest_checkpoint))

def train(epochs, env, batch_size, model, critics, γ, λ, save_dir):
    for i in range(1, epochs + 1):
        start_time = time.time()
        print('Epoch: ', i)
        batch_loss = train_one_epoch(env, batch_size, model, critics, γ, λ, save_dir)
        now = time.time()

        wandb.log({'Epoch': i,
                   'TotalEnvInteracts': i * batch_size,
                   'LossPi': batch_loss,
                   'Time': now - first_start_time})

def test(epochs, env, model):
    for i in range(1, epochs+1):

        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            act, _ = action(model, obs, env)
            obs, rew, done, _ = env.step(act.numpy())
            episode_rew += rew
        print("Episode reward", episode_rew)

class Parser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

first_start_time = time.time()
# training loop

if __name__ == '__main__':
    parser = Parser(description='Train or test PPO')
    parser.add_argument('test', nargs='?', help = 'Test a saved or a random model')
    parser.add_argument('--kl_stop', action='store_true', help= 'Early stopping')
    parser.add_argument('--kl_rollback', action='store_true', help= 'Include early stopping with rollback in the training')
    parser.add_argument('--bootstrap', action='store_true', help='Include bootstrapping when fitting the critic networks')
    parser.add_argument('--norm_rew', action='store_true', help= 'Include Reward Scaling optimization')
    parser.add_argument('--load_dir', help='Optional: directory of saved model to test or resume training')
    parser.add_argument('--env_name', help='Environment name to use with OpenAI Gym')
    parser.add_argument('--save_dir', help='Optional: directory where the model should be saved')
    parser.add_argument('--num_critics', default=3, type=int, help='Number of critics')
    parser.add_argument('--seed', nargs='+', default=[0], type=int, help='Seed')
    parser.add_argument('--wandb_project_name', default='pybullet-GC-experiments4', help='Project name for Weights & Biases experiment tracking')

    args = parser.parse_args()

    env_name = args.env_name
    if(not env_name):
        #parser.error("No env_name provided.")
        env_name="CartPole-v0"

    seeds = args.seed
    n_critics = args.num_critics

    save_dir = args.save_dir
    load_dir = args.load_dir

    batch_size = 10000
    epochs = 200
    learning_rate = 3e-4
    opt = tf.optimizers.Adam(learning_rate)
    γ = .99
    λ = 0.97

    kl_target = 0.01

    for seed in seeds:
        run_name = f"ensemble-{n_critics}-{seed}"
        wandb.init(project=args.wandb_project_name, entity='rlexp', reinit=True, name=run_name, monitor_gym=True, save_code=True)
        wandb.config.env = env_name
        wandb.config.epochs = epochs
        wandb.config.batch_size = batch_size
        wandb.config.learning_rate = learning_rate
        wandb.config.lam = λ
        wandb.config.gamma = γ
        wandb.config.seed = seed
        wandb.config.n_critics = n_critics
        wandb.config.norm_adv = True
        wandb.config.norm_obs = False

        if args.norm_rew:
            wandb.config.norm_rew = True
        else:
            wandb.config.norm_rew = False

        if args.bootstrap:
            wandb.config.bootstrap = True
        else:
            wandb.config.bootstrap = False

        if args.kl_stop or args.kl_rollback:
            wandb.config.kl_target = kl_target
            wandb.config.kl_stop = True
        else:
            wandb.config.kl_stop = False	

        if args.kl_rollback:
            wandb.config.kl_rollback = True
        else:
            wandb.config.kl_rollback = False

        if args.save_dir == None:
            os.makedirs("saves", exist_ok=True)
            save_dir = tempfile.mkdtemp(dir='saves', prefix=run_name)
        else:
            save_dir = args.save_dir


        #environment creation
        env = gym.make(env_name)
        obs_spc = env.observation_space
        act_spc = env.action_space
        if act_spc.shape:
            env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: tf.clip_by_value(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env)
        #env = gym.wrappers.TransformReward(env, lambda reward: tf.clip_by_value(reward, -10, 10))

        #seeding
        tf.random.set_seed(seed)
        env.seed(seed)
        act_spc.seed(seed)
        obs_spc.seed(seed)

        # policy/actor model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(120, activation='relu', input_shape=obs_spc.shape),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(act_spc.shape[0] if act_spc.shape else act_spc.n)
        ])
        if act_spc.shape:
            model.log_std = tf.Variable(tf.fill(env.action_space.shape, -0.5))
        model.summary()

        # value/critic model
        critics = list()

        for _ in range(n_critics):
            value_model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=obs_spc.shape),
                tf.keras.layers.Dense(1)
            ])
            value_model.compile('adam', loss='MSE')
            critics.append(value_model)

        if load_dir:
            load_model(model, load_dir +'/'+ env_name)

        if args.test != None:
            env.render()
            test(epochs, env, model)
        else:
            #env = gym.wrappers.RecordVideo(env, save_dir)
            train(epochs, env, batch_size, model, critics, γ, λ, save_dir)
        wandb.finish()
