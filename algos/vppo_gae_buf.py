import tensorflow as tf
import gym
import pybullet_envs
import time
import math
import argparse
import wandb
import mujoco_py
import tensorflow_probability as tfp

tfd = tfp.distributions

parser = argparse.ArgumentParser(description='train ppo')
parser.add_argument('--env_name', help='environment name')
args = parser.parse_args()
env_name = args.env_name

wandb.init(project='ppo_gae_buf')

env = gym.make(env_name)

obs_spc = env.observation_space
act_spc = env.action_space

batch_size = 5000
epochs = 100
opt = tf.optimizers.Adam(learning_rate=1e-2)
γ = .99
λ = 0.97
eps = 0.1

# config saving
wandb.config.env = env_name
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.lam = λ
wandb.config.gamma = γ

# policy/actor model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=obs_spc.shape),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(act_spc.shape[0] if act_spc.shape else act_spc.n)
])
model.summary()
if act_spc.shape:
    log_std = tf.Variable(tf.fill(env.action_space.shape, -0.5))

# value/critic model
value_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=obs_spc.shape),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])
value_model.compile('adam', loss='MSE')
value_model.summary()


def discount_cumsum(discount_factor, xs):
    # discounts = [1, discount_factor, discount_factor**2, ...]
    discounts = tf.math.cumprod(tf.fill(xs.shape, discount_factor), exclusive=True)
    return tf.math.cumsum(discounts * xs, reverse=True)


class Buffer(object):
    def __init__(self, obs_spc, act_spc, size, gam=0.99, lam=0.97):
        self.ptr = 0
        self.last_idx = 0
        self.size = size
        self.continuous = bool(act_spc.shape)

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
    def finish_path(self, last_val=0):
        current_episode = tf.range(self.last_idx, self.ptr)

        len = self.ptr - self.last_idx
        ret = tf.reduce_sum(self.rew_buf.gather(current_episode)) + last_val
        v_hats = discount_cumsum(self.gam, self.rew_buf.gather(current_episode))

        wandb.log({'EpRet': ret, 'EpLen': len, 'VVals': v_hats.numpy()})

        self.lens.append(len)
        self.rets.append(ret)
        self.V_hats = self.V_hats.scatter(current_episode, v_hats)

        Vs = tf.squeeze(value_model(self.obs_buf.gather(current_episode)), axis=1)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs
        self.gae = self.gae.scatter(current_episode, discount_cumsum(self.gam * self.lam, deltas))

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
        obs, act, adv, logprob = self.obs_buf, self.act_buf, self.gae, self.prob_buf

        if self.continuous:
            dist = tfd.MultivariateNormalDiag(model(obs), tf.exp(log_std))
        else:
            dist = tfd.Categorical(logits=model(obs))

        new_logprob = dist.log_prob(act)

        mask = tf.cast(adv >= 0, tf.float32)
        epsilon_clip = mask * (1 + eps) + (1 - mask) * (1 - eps)
        ratio = tf.exp(new_logprob - logprob)

        return -tf.reduce_mean(tf.minimum(ratio * adv, epsilon_clip * adv))


@tf.function
def action(obs):
    est = tf.squeeze(model(tf.expand_dims(obs, 0)), axis=0)
    if act_spc.shape:
        dist = tfd.MultivariateNormalDiag(est, tf.exp(log_std))
    else:
        dist = tfd.Categorical(logits=est, dtype=act_spc.dtype)

    action = dist.sample()
    logprob = tf.reduce_sum(dist.log_prob(action))

    return action, logprob


def run_one_episode(buf):
    obs = env.reset()
    done = False

    for i in range(buf.ptr, buf.size):
        act, prob = action(obs)
        new_obs, rew, done, _ = env.step(act.numpy())
        buf.store(obs, act, rew, prob)
        obs = new_obs

        if done:
            break

    if done:
        buf.finish_path()
    else:
        buf.finish_path(tf.squeeze(value_model(tf.expand_dims(obs, 0))))


def train_one_epoch():
    batch = Buffer(obs_spc, act_spc, batch_size, gam=γ, lam=λ)
    start_time = time.time()

    while batch.ptr < batch.size:
        run_one_episode(batch)

    train_start_time = time.time()

    loss_fn = batch.loss
    var_list = list(model.trainable_weights)
    if act_spc.shape:
        var_list.append(log_std)

    opt.minimize(loss_fn, var_list=var_list)
    wandb.log({'LossPi': batch.loss()})

    train_time = time.time() - train_start_time
    run_time = train_start_time - start_time

    print('run', run_time, 'train', train_time)
    print('AvgEpRet:', tf.reduce_mean(batch.rets).numpy())

    hist = value_model.fit(batch.obs_buf.numpy(), batch.V_hats.numpy(), batch_size=32)
    wandb.log({'LossV': hist.history['loss']})

    return batch.rets, batch.lens


first_start_time = time.time()
# training loop
for i in range(epochs):
    start_time = time.time()
    batch_rets, batch_lens = train_one_epoch()
    duration = time.time() - start_time
    now = time.time()

    wandb.log({'Epoch': i, 'TotalEnvInteracts': i * batch_size, 'Time': now - first_start_time})