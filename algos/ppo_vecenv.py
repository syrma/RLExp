import tensorflow as tf
import wandb
import argparse
import sys
import gym
import pybullet_envs
import time
import math
import gym_vecenv
import tensorflow_probability as tfp

tfd = tfp.distributions

@tf.function
def action(model, obs, env):
    est = model(obs)
    if env.action_space.shape:
        dist = tfd.MultivariateNormalDiag(est, tf.exp(model.log_std))
    else:
        dist = tfd.Categorical(logits=est, dtype=env.action_space.dtype)

    action = dist.sample()
    logprob = tf.reduce_sum(dist.log_prob(action))

    return action, logprob

def loss(obs, act, adv, logprob):
    eps = 0.1

    if act_spc.shape:
        dist = tfd.MultivariateNormalDiag(model(obs), tf.exp(model.log_std))
    else:
        dist = tfd.Categorical(logits=model(obs))

    new_logprob = tf.reduce_sum(dist.log_prob(act), axis=1)

    masks = [tf.cast(adv[i] >= 0, tf.float32) for i in range(num_env)]

    tf.math.multiply(masks, eps)

    epsilon_clip = [
        masks[i] * (1 + eps) + (1 - masks[i]) * (1 - eps) for i in range(num_env)
    ]

    ratio = tf.exp(new_logprob - logprob)

    s1 = tf.math.multiply(ratio, adv)
    s2 = tf.math.multiply(epsilon_clip, adv)

    return -tf.reduce_mean(tf.minimum(s1, s2))

def run_env(env, size, model, value_model, γ, λ):

    obs_buf = tf.TensorArray(obs_spc.dtype, size)
    act_buf = tf.TensorArray(act_spc.dtype, size)
    rew_buf = tf.TensorArray(tf.float32, size)
    prob_buf = tf.TensorArray(tf.float32, size)
    done_buf = tf.TensorArray(tf.bool, size)

    obs = env.reset()
    obs = tf.cast(obs, obs_spc.dtype)

    for i in range(size):
        act, prob = action(model, obs, env)
        new_obs, rew, done, _ = env.step(act.numpy())

        obs_buf = obs_buf.write(i, obs)
        act_buf = act_buf.write(i, act)
        rew_buf = rew_buf.write(i, rew)
        prob_buf = prob_buf.write(i, prob)
        done_buf = done_buf.write(i, done)

        obs = tf.cast(new_obs, obs_spc.dtype)

    obs_buf = obs_buf.stack()
    act_buf = act_buf.stack()
    rew_buf = rew_buf.stack()
    prob_buf = prob_buf.stack()
    done_buf = done_buf.stack()

    # last_val is 0 when done
    last_val = tf.where(done_buf[-1], 0, tf.squeeze(value_model(obs)))

    return obs_buf, act_buf, rew_buf, prob_buf, done_buf, last_val


#@tf.function
def critic(obs_buf, rew_buf, done_buf, last_val, γ, λ):
    rets = []
    # lens = []

    # TODO: turn into a list of tensor arrays
    v_hats = [tf.TensorArray(tf.float32, size) for _ in range(num_env)]
    gae = [tf.TensorArray(tf.float32, size) for _ in range(num_env)]

    # TODO: changer la boucle et remplacer cumprod/cumsum
    last_idx = [0] * num_env

    for i in range(size):
        for j in range(num_env):  # num_env = ?
            if i != size - 1 and not done_buf[i, j]:
                continue

            # sum of discounted rewards
            current_episode = slice(last_idx[j], i + 1)
            ep_idx = range(last_idx[j], i + 1)
            ep_rew = rew_buf[current_episode, j]
            # TODO: le premier argument de tf.fill n'est pas connu à l'avance (ne marche pas avec tf.function)
            discounts = tf.math.cumprod(tf.fill(ep_rew.shape, γ), exclusive=True)
            ep_v_hats = tf.math.cumsum(discounts * ep_rew, reverse=True)
            v_hats[j] = v_hats[j].scatter(ep_idx, ep_v_hats)

            Vs = tf.squeeze(value_model(obs_buf[current_episode, j]), axis=1)
            if i == size - 1:
                Vsp1 = tf.concat([Vs[1:], [last_val[j]]], axis=0)
            else:
                Vsp1 = tf.concat([Vs[1:], [0]], axis=0)

            deltas = rew_buf[current_episode, j] + γ * Vsp1 - Vs

            # compute the advantage function (gae)
            discounts = tf.math.cumprod(tf.fill(deltas.shape, γ * λ), exclusive=True)
            ep_gae = tf.math.cumsum(discounts * deltas, reverse=True)
            gae[j] = gae[j].scatter(ep_idx, ep_gae)

            if i == size - 1:
                rets.append(tf.reduce_sum(rew_buf[current_episode, j]) + last_val[j])
            else:
                rets.append(tf.reduce_sum(rew_buf[current_episode, j]))
            last_idx[j] = i + 1

    v_hats = [v_hat.stack() for v_hat in v_hats]
    gae = [g.stack() for g in gae]

    return rets, v_hats, gae
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
    var_list = list(model.trainable_weights)
    if env.action_space.shape:
        var_list.append(model.log_std)

    for i in range(epochs):
        start_time = time.time()
        print('Epoch: ', i)

        obs_buf, act_buf, rew_buf, prob_buf, done_buf, last_val = run_env(env, size, model, value_model, γ, λ)

        end_run_time = time.time()

        rets, v_hats, adv = critic(obs_buf, rew_buf, done_buf, last_val, γ, λ)

        train_start_time = time.time()

        opt.minimize(lambda: loss(obs_buf, act_buf, adv, prob_buf), var_list=var_list)
        hist = value_model.fit(obs_buf, v_hats, batch_size=32)

        run_time = end_run_time - start_time
        critic_time = train_start_time - end_run_time
        train_time = time.time() - train_start_time

        print("run time", run_time, "critic time", critic_time, "train time", train_time)
        # print(f"run time: {run_time}, critic time: {critic_time}, train time: {train_time}")
        print("AvgEpRet:", tf.reduce_mean(rets).numpy())
        if tf.reduce_mean(rets).numpy() > 200:
            break

        wandb.log({'LossV': tf.reduce_mean(hist.history['loss']).numpy(),
                   'EpRet': wandb.Histogram(rets),
                   'AvgEpRet': tf.reduce_mean(rets),
                    #'EpLen': tf.reduce_mean(lens),
                   'VVals': wandb.Histogram(v_hats),
                   'Epoch': i,
                   'TotalEnvInteracts': i * batch_size,
                   'RunTime': run_time,
                   'TrainTime': train_time,
                   'CriticTime': critic_time}
                  )


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
    save_dir = args.save_dir if args.save_dir else "model/"
    load_dir = args.load_dir
    num_runs = int(args.num_runs) if args.num_runs else 1

    size = 5000
    epochs = 200
    learning_rate = 1e-2
    opt = tf.optimizers.Adam(learning_rate)
    γ = .99
    λ = 0.97
    num_env = 5

    env = gym.make(env_name)
    env = gym_vecenv.DummyVecEnv([lambda: gym.make(env_name)] * num_env)

    obs_spc = env.observation_space
    act_spc = env.action_space

    for x in range(num_runs):
        wandb.init(project='ppo_vecenv_oldarch', entity='rlexp', reinit=True)
        wandb.config.env = env_name
        wandb.config.epochs = epochs
        wandb.config.size = size
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
            load_model(model, load_dir + env_name)

        if args.test != None:
            env.render()
            test(epochs, env, model)
        else:
            train(epochs, env, size, model, value_model, γ, λ)
            if save_dir==None:
                save_dir = 'model/'
                save_model(model, save_dir + env_name)
        wandb.finish()
