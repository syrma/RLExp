import os

import tensorflow as tf
import gym
import pybullet_envs
import time
import math
import argparse
import wandb
import sys
from gym.wrappers import Monitor
import tempfile
import tensorflow_probability as tfp
tfd = tfp.distributions

class Buffer(object):
    def __init__(self, obs_spc, act_spc, model, value_model, size, gam=0.99, lam=0.97):
        self.ptr = 0
        self.last_idx = 0
        self.size = size
        self.continuous = bool(act_spc.shape)

        self.model = model
        self.value_model = value_model

        self.obs_buf = tf.TensorArray(obs_spc.dtype, size)
        self.act_buf = tf.TensorArray(act_spc.dtype, size)
        self.rew_buf = tf.TensorArray(tf.float32, size)
        self.prob_buf = tf.TensorArray(tf.float32, size)

        self.rets = []
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
        last_val = tf.squeeze(self.value_model(tf.expand_dims(last_obs, 0))) if last_obs is not None else 0

        length = self.ptr - self.last_idx
        ret = tf.reduce_sum(self.rew_buf.gather(current_episode)) + last_val

        # v_hats = discounted cumulative sum
        ep_rew = self.rew_buf.gather(current_episode)
        discounts = tf.math.cumprod(tf.fill(ep_rew.shape, self.gam), exclusive=True)
        v_hats = tf.math.cumsum(discounts * ep_rew, reverse=True)

        self.lens.append(length)
        self.rets.append(ret)
        self.V_hats = self.V_hats.scatter(current_episode, v_hats)

        Vs = tf.squeeze(value_model(self.obs_buf.gather(current_episode)), axis=1)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs

        # compute the advantage function (gae)
        discounts = tf.math.cumprod(tf.fill(deltas.shape, self.gam * self.lam), exclusive=True)
        gae = tf.math.cumsum(discounts * deltas, reverse=True)
        self.gae = self.gae.scatter(current_episode, gae)

        self.last_idx = self.ptr

        if self.ptr == self.size:
            self.obs_buf = self.obs_buf.stack()
            self.act_buf = self.act_buf.stack()
            self.rew_buf = self.rew_buf.stack()
            self.prob_buf = self.prob_buf.stack()

            self.V_hats = self.V_hats.stack()
            self.gae = self.gae.stack()

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

        buf.store(obs, act, rew, prob)
        obs = tf.cast(new_obs, obs_dtype)

        if done:
            break

    critic_start = time.time()
    if done:
        buf.finish_path()
    else:
        while not done:
            act, prob = action(buf.model, obs, env)
            new_obs, rew, done, _ = env.step(act.numpy())
        buf.finish_path(obs)

    return time.time() - critic_start


def train_one_epoch(env, batch_size, model, value_model, γ, λ):
    obs_spc = env.observation_space
    act_spc = env.action_space

    batch = Buffer(obs_spc, act_spc, model, value_model, batch_size, gam=γ, lam=λ)
    start_time = time.time()

    critic_time = 0
    while batch.ptr < batch.size:
        critic_time += run_one_episode(env, batch)

    train_start_time = time.time()

    var_list = list(model.trainable_weights)
    if act_spc.shape:
        var_list.append(model.log_std)

    opt.minimize(batch.loss, var_list=var_list)

    train_time = time.time() - train_start_time
    run_time = train_start_time - start_time

    print('run time', run_time, 'critic time (included in run time):', critic_time, 'train time', train_time)
    print('AvgEpRet:', tf.reduce_mean(batch.rets).numpy())

    hist = value_model.fit(batch.obs_buf.numpy(), batch.V_hats.numpy(), batch_size=32)
    wandb.log({'LossV': tf.reduce_mean(hist.history['loss']).numpy(),
               'EpRet': wandb.Histogram(batch.rets),
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

def train(epochs, env, batch_size, model, value_model, γ, λ):
    for i in range(1, epochs + 1):
        start_time = time.time()
        print('Epoch: ', i)
        batch_loss = train_one_epoch(env, batch_size, model, value_model, γ, λ)
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
    parser.add_argument('--load_dir', help='Optional: directory of saved model to test or resume training')
    parser.add_argument('--env_name', help='Environment name to use with OpenAI Gym')
    parser.add_argument('--save_dir', help='Optional: directory where the model should be saved')
    parser.add_argument('--num_runs', help='Number of runs')

    args = parser.parse_args()

    env_name = args.env_name
    if(not env_name):
        #parser.error("No env_name provided.")
        env_name="CartPole-v0"

    save_dir = args.save_dir
    load_dir = args.load_dir
    num_runs = int(args.num_runs) if args.num_runs else 1

    env = gym.make(env_name)
    obs_spc = env.observation_space
    act_spc = env.action_space

    batch_size = 5000
    epochs = 30
    learning_rate = 1e-2
    opt = tf.optimizers.Adam(learning_rate)
    γ = .99
    λ = 0.97

    for x in range(num_runs):
        exp_name = "ppo-" + env_name + str(time.time())
        wandb.init(project='ppo', entity='rlexp', reinit=True, name=exp_name, monitor_gym=True, save_code=True)
        wandb.config.env = env_name
        wandb.config.epochs = epochs
        wandb.config.batch_size = batch_size
        wandb.config.learning_rate = learning_rate
        wandb.config.lam = λ
        wandb.config.gamma = γ

        # policy/actor model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', input_shape=obs_spc.shape),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(act_spc.shape[0] if act_spc.shape else act_spc.n)
        ])
        if act_spc.shape:
            model.log_std = tf.Variable(tf.fill(env.action_space.shape, -0.5))
        model.summary()

        # value/critic model
        value_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', input_shape=obs_spc.shape),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])
        value_model.compile('adam', loss='MSE')
        value_model.summary()

        if load_dir:
            load_model(model, load_dir +'/'+ env_name)

        recordings = tempfile.mkdtemp(prefix='recordings', dir='.')
        if args.test != None:
            env.render()
            test(epochs, env, model)
        else:
            monitor_env = Monitor(env, recordings, force=True)
            train(epochs, env, batch_size, model, value_model, γ, λ)
            if save_dir==None:
                save_dir = 'model/'
                save_model(model, save_dir+env_name)
        
        wandb.finish()
