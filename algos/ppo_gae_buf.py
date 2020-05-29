import tensorflow as tf
import gym
import pybullet_envs
import time
import math
import argparse
import wandb

parser = argparse.ArgumentParser(description='train ppo')
parser.add_argument('--env_name', help='environment name')
args = parser.parse_args()
env_name = args.env_name

wandb.init(project='ppo_gae_buf')
save_freq = 10

env = gym.make(env_name)
obs_shape = env.observation_space.shape
n_acts = env.action_space.shape[0]

batch_size = 5000
epochs = 100
opt = tf.optimizers.Adam(learning_rate=1e-2)
γ = [0.99, 0.99]
λ = [0.97, 0.99]
eps = 0.1

#config saving
wandb.config.env = env_name
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.lam = λ
wandb.config.gamma = γ

#policy/actor model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='tanh', input_shape=obs_shape),
    tf.keras.layers.Dense(env.action_space.shape[0])
])
model.summary()
log_std = tf.Variable(tf.fill(env.action_space.shape, -0.5))

#value/critic model
value_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='tanh', input_shape=obs_shape),
    tf.keras.layers.Dense(1)
])
value_model.compile('adam', loss='MSE')
value_model.summary()

def discount_cumsum(discount_factor, xs):
    # discounts = [1, discount_factor, discount_factor**2, ...]
    discounts = tf.math.cumprod(tf.fill(xs.shape, discount_factor), exclusive=True)
    return tf.math.cumsum(discounts * xs, reverse=True)

class Buffer(object):
    def __init__(self, obs_shape, n_acts, size, gam=0.99, lam=0.97):
        self.ptr = 0
        self.last_idx = 0
        self.size = size
        self.n_acts = n_acts

        self.obs_buf = tf.TensorArray(tf.float32, size)
        self.act_buf = tf.TensorArray(tf.float32, size) # a changé. si discret int, si box float
        self.rew_buf = tf.TensorArray(tf.float32, size)
        self.prob_buf = tf.TensorArray(tf.float32,  size)


        self.rets = []
        self.lens = []

        self.V_hats = tf.TensorArray(tf.float32, size)
        self.n_gae = 2
        self.gae = [tf.TensorArray(tf.float32, size) for _ in range(self.n_gae)]

        self.gam = gam
        self.lam = lam

    #@tf.function
    def store(self, obs, act, rew, prob):
        self.obs_buf = self.obs_buf.write(self.ptr, obs)
        self.act_buf = self.act_buf.write(self.ptr, act)
        self.rew_buf = self.rew_buf.write(self.ptr, rew)
        self.prob_buf = self.prob_buf.write(self.ptr, prob)
        self.ptr += 1

    #@tf.function
    def finish_path(self, last_val=0):
        current_episode = tf.range(self.last_idx, self.ptr)

        len = self.ptr - self.last_idx
        ret = tf.reduce_sum(self.rew_buf.gather(current_episode)) + last_val
        v_hats = discount_cumsum(self.gam, self.rew_buf.gather(current_episode))

        wandb.log({'EpRet':ret, 'EpLen':len,'VVals':v_hats.numpy()})

        self.lens.append(len)
        self.rets.append(ret)
        self.V_hats = self.V_hats.scatter(current_episode, v_hats)

        Vs = tf.squeeze(value_model.apply(self.obs_buf.gather(current_episode)), axis=1)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs
        self.gae[i] = self.gae[i].scatter(current_episode, discount_cumsum(self.gam[i] * self.lam[i], deltas))

        self.last_idx = self.ptr

        if self.ptr == self.size:
          self.obs_buf = self.obs_buf.stack()
          self.act_buf = self.act_buf.stack()
          self.rew_buf = self.rew_buf.stack()
          self.prob_buf = self.prob_buf.stack()

          self.V_hats = self.V_hats.stack()
          self.gae = self.gae.stack()

    #@tf.function
    def loss(self, minibatch_start, minibatch_size):
        minibatch = slice(minibatch_start, minibatch_start + minibatch_size)
        obs, act, adv, logprob = self.obs_buf[minibatch], self.act_buf[minibatch], self.prob_buf[minibatch]

        adv = (self.gae[0][minibatch] + self.gae[1][minibatch]) / 2
        minibatch_size = len(adv)

        #logits = model(obs)
              
        # # forall i, newprob[i] = softmax(logits)[i, act[i]]
        # # softmax(logits).shape = (minibatch_size, n_acts)
        #new_prob = tf.gather_nd(tf.nn.softmax(logits), tf.reshape(act, (minibatch_size, 1)), batch_dims=1)
        
        mu = model(obs)
        new_logprob = gaussian_likelihood(act, mu, log_std)


        mask = tf.cast(adv >= 0, tf.float32)
        epsilon_clip = mask * (1 + eps) + (1 - mask) * (1 - eps)
        return -tf.reduce_mean(tf.minimum(tf.exp(new_logprob - logprob) * adv, epsilon_clip * adv))

EPS = 1e-8
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + tf.math.log(2*math.pi))
    return tf.reduce_sum(pre_sum, axis=1)

@tf.function
def action(obs):
      mu = model(tf.expand_dims(obs, 0))
      std = tf.exp(log_std)
      #action = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=(0,1))
      action = mu + tf.random.normal(env.action_space.shape) * std 
#     prob = tf.nn.softmax(logits)[0,action]
      prob = gaussian_likelihood(action, mu, log_std)
      return action, prob

def run_one_episode(buf):
    obs = env.reset()
    done = False

    for i in range(buf.ptr, buf.size):
        act, prob = action(obs)
        new_obs, rew, done, _ = env.step(act[0].numpy())
        buf.store(obs, act[0], rew, prob)
        obs = new_obs

        if done:
            break

    if done:
        buf.finish_path()
    else:
        buf.finish_path(tf.squeeze(value_model(tf.expand_dims(obs, 0))))

def train_one_epoch():

    batch = Buffer(obs_shape, n_acts, batch_size, gam=γ, lam=λ)
    start_time = time.time()
    
    while batch.ptr < batch.size:
        run_one_episode(batch)
    
    train_start_time = time.time()
    
    minibatch_size = 32
    for minibatch_start in range(0, batch.size, minibatch_size):
        opt.minimize(lambda: batch.loss(minibatch_start, minibatch_size), var_list=model.trainable_weights + [log_std])
        wandb.log({'LossPi':batch.loss(minibatch_start, minibatch_size)})
    
    train_time = time.time() - train_start_time
    run_time = train_start_time - start_time

    print('run', run_time, 'train', train_time)

    hist = value_model.fit(batch.obs_buf.numpy(), batch.V_hats.numpy(), batch_size=minibatch_size)
    wandb.log({'LossV':hist.history['loss']})

    return batch.rets, batch.lens

first_start_time = time.time()
#training loop
for i in range(epochs):
    start_time = time.time()
    batch_rets, batch_lens = train_one_epoch()
    duration = time.time() - start_time
    now = time.time()

    wandb.log({'Epoch': i, 'TotalEnvInteracts': i * batch_size, 'Time': now - first_start_time})