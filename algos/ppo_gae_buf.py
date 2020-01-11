import tensorflow as tf
import gym
from baselines import logger
import time

env = gym.make('Walker2d-v2')
obs_shape = env.observation_space.shape
n_acts = env.action_space.shape[0]

epochs = 100
batch_size = 5000
opt = tf.optimizers.Adam(learning_rate=1e-2)
γ = .99
λ = 0.97
eps = 0.1

#policy/actor model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='tanh', input_shape=obs_shape),
    tf.keras.layers.Dense(n_acts)
])
model.summary()

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
    def __init__(self, obs_shape, n_acts, size=5000, gam=0.99, lam=0.97):
        self.ptr = 0
        self.last_idx = 0
        self.size = size
        self.n_acts = n_acts

        self.obs_buf = tf.TensorArray(tf.float32, size)
        self.act_buf = tf.TensorArray(tf.int64, size)
        self.rew_buf = tf.TensorArray(tf.float32, size)
        self.prob_buf = tf.TensorArray(tf.float32, size)

        self.rets = []
        self.lens = []

        self.V_hats = tf.TensorArray(tf.float32, size)
        self.gae = tf.TensorArray(tf.float32, size)

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
        self.lens.append(self.ptr - self.last_idx)
        self.rets.append(tf.reduce_sum(self.rew_buf.gather(current_episode)) + last_val)

        self.V_hats = self.V_hats.scatter(current_episode, discount_cumsum(self.gam, self.rew_buf.gather(current_episode)))

        Vs = tf.squeeze(value_model(self.obs_buf.gather(current_episode)), axis=1)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs

        self.gae = self.gae.scatter(current_episode, discount_cumsum(self.gam * self.lam, deltas))

        self.last_idx = self.ptr
        if self.ptr==self.size:
            self.obs_buf = self.obs_buf.stack()
            self.act_buf = self.act_buf.stack()
            self.rew_buf = self.rew_buf.stack()
            self.prob_buf = self.prob_buf.stack()

            self.V_hats = self.V_hats.stack()
            self.gae = self.gae.stack()

    #@tf.function
    def loss(self, minibatch_start, minibatch_size):
        minibatch = slice(minibatch_start, minibatch_start + minibatch_size)
        obs, act, adv, prob = self.obs_buf[minibatch], self.act_buf[minibatch], self.gae[minibatch], self.prob_buf[minibatch]
        minibatch_size = len(adv)

        logits = model(obs)
        mask = tf.cast(adv >= 0, tf.float32)
        epsilon_clip = mask * (1 + eps) + (1 - mask) * (1 - eps)
        # forall i, newprob[i] = softmax(logits)[i, act[i]]
        # softmax(logits).shape = (minibatch_size, n_acts)
        new_prob = tf.gather_nd(tf.nn.softmax(logits), tf.reshape(act, (minibatch_size, 1)), batch_dims=1)

        return -tf.reduce_mean(tf.minimum(new_prob/prob * adv, epsilon_clip * adv))

@tf.function
def action(obs):
    logits = model(tf.reshape(obs, (1,-1)))
    action = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=(0,1))
    prob = tf.nn.softmax(logits)[0,action]
    return action, prob

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
        buf.finish_path(tf.squeeze(value_model.apply(obs.reshape(1, -1))))

def train_one_epoch():

    batch = Buffer(obs_shape, n_acts, batch_size, gam=γ, lam=λ)
    start_time = time.time()
    
    while batch.ptr < batch.size:
        run_one_episode(batch)
    
    train_start_time = time.time()
    
    minibatch_size = 32
    for minibatch_start in range(0, batch.size, minibatch_size):
        opt.minimize(lambda: batch.loss(minibatch_start, minibatch_size), var_list=model.trainable_weights)
    
    train_time = time.time() - train_start_time
    run_time = train_start_time - start_time

    print('run', run_time, 'train', train_time)
    value_model.fit(batch.obs_buf.numpy(), batch.V_hats.numpy(), batch_size=minibatch_size)

    return batch.rets, batch.lens

#training loop

first_start_time = time.time()
for i in range(epochs):
    start_time = time.time()
    batch_rets, batch_lens = train_one_epoch()
    now = time.time()

    logger.logkv("misc/nupdates", i)
    logger.logkv("misc/total_timesteps", i*batch_size)
    logger.logkv("fps", int(batch_size/(now - start_time)))
    logger.logkv("eprewmean", tf.reduce_mean(batch_rets).numpy())
    logger.logkv("eplenmean", tf.reduce_mean(batch_lens).numpy())
    logger.logkv("elapsed_time", now - first_start_time)
    #logger.logkv("loss/", batch_loss.numpy())

    logger.dumpkvs()