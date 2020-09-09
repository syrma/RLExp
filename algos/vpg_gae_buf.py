import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import gym
import pybullet_envs
import time
import wandb
import math

env_name = 'AntBulletEnv-v0'

env = gym.make(env_name)

obs_spc = env.observation_space
act_spc = env.action_space

epochs = 100
batch_size = 5000
opt = tf.optimizers.Adam(learning_rate=1e-2)
γ = .99
λ = 0.97

wandb.init('RLExp-algos', entity='rlexp')
wandb.config.env = env_name
wandb.config.algo = 'vpg_gae_buf'
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.lam = λ
wandb.config.gamma = γ

# construct the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=obs_spc.shape),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(act_spc.shape[0] if act_spc.shape else act_spc.n)
])
if act_spc.shape:
    log_std = tf.Variable(tf.zeros(act_spc.shape))
model.summary()

value_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=obs_spc.shape),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])
value_model.compile('adam', loss='MSE')
value_model.summary()

@tf.function(experimental_relax_shapes=True)
def discount_cumsum(discount_factor, xs, length):
    #print(type(discount_factor), type(xs), xs.shape)
    # discounts = [1, discount_factor, discount_factor**2, ...]
    discounts = tf.math.cumprod(tf.fill(length, discount_factor), exclusive=True)
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
    def finish_path(self, last_val=0):
        current_episode = range(self.last_idx, self.ptr)

        length = self.ptr - self.last_idx
        ret = tf.reduce_sum(self.rew_buf.gather(current_episode)) + last_val
        ep_rew = self.rew_buf.gather(current_episode)
        v_hats = discount_cumsum(self.gam, ep_rew, tf.constant([length]))

        self.lens.append(length)
        self.rets.append(ret)
        self.V_hats = self.V_hats.scatter(current_episode, v_hats)

        Vs = tf.squeeze(value_model(self.obs_buf.gather(current_episode)), axis=1)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs

        self.gae = self.gae.scatter(current_episode, discount_cumsum(self.gam * self.lam, deltas, tf.constant([length])))

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

            dist = tfd.Normal(model(self.obs_buf), tf.exp(log_std))

        else: # Discrete
            dist = tfd.Categorical(logits=model(self.obs_buf))

        log_probs = dist.log_prob(self.act_buf)
        #qprint(self.act_buf.shape, self.gae.shape, log_probs.shape)
        return -tf.reduce_mean(self.gae * log_probs)

#@tf.function
def action(obs):
    est = tf.squeeze(model(tf.expand_dims(obs, 0)), axis=0)
    if act_spc.shape: # Box
        dist = tfd.Normal(est, tf.exp(log_std))
    else: # Discrete
        dist = tfd.Categorical(logits=est, dtype=act_spc.dtype)

    return dist.sample()

def run_one_episode(buf):
    obs = env.reset()
    obs = tf.cast(obs, obs_spc.dtype)
    done = False

    for i in range(buf.ptr, buf.size):
        act = action(obs)
        new_obs, rew, done, _ = env.step(act.numpy())

        buf.store(obs, act, rew)
        obs = new_obs
        obs = tf.cast(obs, obs_spc.dtype)

        if done:
            break

    if done:
        buf.finish_path()
    else:
        buf.finish_path(tf.squeeze(value_model(tf.expand_dims(obs, 0))))

def train_one_epoch():

    batch = Buffer(obs_spc, act_spc, batch_size, gam=γ, lam=λ)

    while batch.ptr < batch.size:
        run_one_episode(batch)

    var_list = list(model.trainable_weights)
    if act_spc.shape:
        var_list.append(log_std)

    opt.minimize(batch.loss, var_list=var_list)

    #print('batch_obs: {}, batch_V_hats: {}'.format(batch.obs.shape,
    #                                               batch.V_hats.shape))
    hist = value_model.fit(batch.obs_buf.numpy(), batch.V_hats.numpy())
    wandb.log({'LossV': tf.reduce_mean(hist.history['loss']).numpy(),
               'EpRet': wandb.Histogram(batch.rets),
               'AvgEpRet': tf.reduce_mean(batch.rets),
               'EpLen': tf.reduce_mean(batch.lens),
               'VVals': wandb.Histogram(batch.V_hats)},
              commit=False)
    print('AvgEpRet:', tf.reduce_mean(batch.rets).numpy())
    return batch.loss()

first_start_time = time.time()

def save_model():
    save_path = ("./model")
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=None)
    manager.save()

#training loop
for i in range(1, epochs+1):

    start_time = time.time()
    batch_loss = train_one_epoch()
    now = time.time()

    wandb.log({'Epoch': i,
               'TotalEnvInteracts': i*batch_size,
               'LossPi': batch_loss,
               'Time': now - first_start_time})

#save_model()
