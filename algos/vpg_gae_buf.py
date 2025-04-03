import tensorflow as tf
import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
import gym
import pybullet_envs
import time
import wandb
import math
import argparse
import tempfile

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

        self.rets = []
        self.lens = []

        self.V_hats = tf.TensorArray(tf.float32, size)
        self.gae = tf.TensorArray(tf.float32, size)

        self.gam = gam
        self.lam = lam

    def store(self, obs, act, rew):
        self.obs_buf = self.obs_buf.write(self.ptr, obs)
        self.act_buf = self.act_buf.write(self.ptr, act)
        self.rew_buf = self.rew_buf.write(self.ptr, rew)
        self.ptr += 1

    #@tf.function
    def finish_path(self, last_obs=None):
        current_episode = range(self.last_idx, self.ptr)
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

        Vs = tf.squeeze(self.value_model(self.obs_buf.gather(current_episode)), axis=1)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs

        # compute the advantage function (gae)
        discounts = tf.math.cumprod(tf.fill(deltas.shape, self.gam * self.lam), exclusive=True)
        gae = tf.math.cumsum(discounts * deltas, reverse=True)
        self.gae = self.gae.scatter(current_episode, gae)

        self.last_idx = self.ptr
        if self.ptr==self.size:
            self.obs_buf = self.obs_buf.stack()
            self.act_buf = self.act_buf.stack()
            self.rew_buf = self.rew_buf.stack()

            self.V_hats = self.V_hats.stack()
            self.gae = self.gae.stack()

    #@tf.function
    def loss(self):
        if self.continuous: # Box
            # π = N(μ, σ) with μ=model(obs), σ
            dist = tfd.MultivariateNormalDiag(self.model(self.obs_buf), tf.exp(self.model.log_std))

        else: # Discrete
            dist = tfd.Categorical(logits=self.model(self.obs_buf))

        log_probs = dist.log_prob(self.act_buf)
        return -tf.reduce_mean(self.gae * log_probs)

#@tf.function
def action(model, obs, env):
    est = tf.squeeze(model(tf.expand_dims(obs, 0)), axis=0)
    if env.action_space.shape: # Box
        dist = tfd.MultivariateNormalDiag(est, tf.exp(model.log_std))
    else: # Discrete
        dist = tfd.Categorical(logits=est, dtype=env.action_space.dtype)

    return dist.sample()

def run_one_episode(env, buf):
    obs_dtype = env.observation_space.dtype

    obs = env.reset()
    obs = tf.cast(obs, obs_dtype)
    done = False

    for i in range(buf.ptr, buf.size):
        act = action(buf.model, obs, env)
        new_obs, rew, done, _ = env.step(act.numpy())

        buf.store(obs, act, rew)
        obs = new_obs
        obs = tf.cast(obs, obs_dtype)

        if done:
            break

    if done:
        buf.finish_path()
    else:
        while not done:
            act = action(buf.model, obs, env)
            new_obs, rew, done, _ = env.step(act.numpy())
        buf.finish_path(obs)

def train_one_epoch(env, batch_size, model, value_model, γ, λ):
    obs_spc = env.observation_space
    act_spc = env.action_space

    batch = Buffer(obs_spc, act_spc, model, value_model, batch_size, gam=γ, lam=λ)
    start_time = time.time()

    while batch.ptr < batch.size:
        run_one_episode(env, batch)

    train_start_time = time.time()

    var_list = list(model.trainable_weights)
    if act_spc.shape:
        var_list.append(model.log_std)

    with tf.GradientTape() as tape:
        loss = batch.loss()

    grads = tape.gradient(loss, var_list)
    opt.apply(grads, trainable_variables=var_list)

    train_time = time.time() - train_start_time
    run_time = train_start_time - start_time

    print('run time', run_time, 'train time', train_time)
    print('AvgEpRet:', tf.reduce_mean(batch.rets).numpy())

    hist = value_model.fit(batch.obs_buf.numpy(), batch.V_hats.numpy())
    wandb.log({'LossV': tf.reduce_mean(hist.history['loss']).numpy(),
               'EpRet': wandb.Histogram(batch.rets),
               'AvgEpRet': tf.reduce_mean(batch.rets),
               'EpLen': tf.reduce_mean(batch.lens),
               'VVals': wandb.Histogram(batch.V_hats)},
              commit=False)

    return batch.loss()

first_start_time = time.time()

def train(epochs, env, batch_size, model, value_model, γ, λ):
    for i in range(1, epochs+1):
        start_time = time.time()
        print('Epoch', i)
        batch_loss = train_one_epoch(env, batch_size, model, value_model, γ, λ)
        now = time.time()
        
        wandb.log({'Epoch': i,
                   'TotalEnvInteracts': i*batch_size,
                   'LossPi': batch_loss,
                   'Time': now - first_start_time})

def test(epochs, env, model):
    for i in range(1, epochs+1):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            act = action(model, obs, env.action_space)
            obs, rew, done, _ = env.step(act.numpy())
            episode_rew += rew
        print("Episode reward", episode_rew)

if __name__=="__main__":
    num_runs = 1
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    obs_spc = env.observation_space
    act_spc = env.action_space

    epochs = 100
    batch_size = 5000
    learning_rate = 1e-2
    opt = keras.optimizers.Adam(learning_rate)
    γ = .99
    λ = 0.97

    for x in range(num_runs):
        exp_name = "vpg-" + env_name + str(time.time())
        wandb.init(project='vpg', entity='rlexp', reinit=True, name='new_architecture', monitor_gym=True, save_code=True)
        wandb.config.env = env_name
        wandb.config.algo = 'vpg_gae_buf'
        wandb.config.epochs = epochs
        wandb.config.batch_size = batch_size
        wandb.config.learning_rate = learning_rate
        wandb.config.lam = λ
        wandb.config.gamma = γ

        # construct the model
        model = keras.models.Sequential([
            keras.layers.Dense(120, activation='relu', input_shape=obs_spc.shape),
            keras.layers.Dense(84, activation='relu'),
            keras.layers.Dense(act_spc.shape[0] if act_spc.shape else act_spc.n)
        ])
        if act_spc.shape:
            model.log_std = tf.Variable(tf.zeros(act_spc.shape))
        model.summary()

        value_model = keras.models.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=obs_spc.shape),
            keras.layers.Dense(1)
        ])
        value_model.compile('adam', loss='MSE')
        value_model.summary()

        with tempfile.TemporaryDirectory(prefix='recordings', dir='.') as recordings:
            train(epochs, env, batch_size, model, value_model, γ, λ)

        wandb.finish()
