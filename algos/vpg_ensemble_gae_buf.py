import tensorflow as tf
import gym

import time

env = gym.make('CartPole-v0')
obs_shape = env.observation_space.shape
n_acts = env.action_space.n

epochs = 100
opt = tf.optimizers.Adam(learning_rate=1e-2)
γ = .99
λ = 0.97

# construct the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='tanh', input_shape=obs_shape),
    tf.keras.layers.Dense(n_acts)
])
model.summary()

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

        self.rets = []
        self.lens = []

        self.V_hats = tf.TensorArray(tf.float32, size)
        self.gae = tf.TensorArray(tf.float32, size)

        self.gam = gam
        self.lam = lam

    #@tf.function
    def store(self, obs, act, rew):
        self.obs_buf = self.obs_buf.write(self.ptr, obs)
        self.act_buf = self.act_buf.write(self.ptr, act)
        self.rew_buf = self.rew_buf.write(self.ptr, rew)
        self.ptr += 1

    #@tf.function
    def finish_path(self, last_val=0):
        current_episode = range(self.last_idx, self.ptr)
        self.lens.append(self.ptr - self.last_idx)
        self.rets.append(tf.reduce_sum(self.rew_buf.gather(current_episode)) + last_val)

        self.V_hats.scatter(current_episode, discount_cumsum(self.gam, self.rew_buf.gather(current_episode)))

        Vs = tf.squeeze(value_model(self.obs_buf.gather(current_episode)), axis=1)
        Vsp1 = tf.concat([Vs[1:], [last_val]], axis=0)
        deltas = self.rew_buf.gather(current_episode) + self.gam * Vsp1 - Vs

        self.gae.scatter(current_episode, discount_cumsum(self.gam * self.lam, deltas))

        self.last_idx = self.ptr
        if self.ptr==self.size:
            self.obs_buf = self.obs_buf.stack()
            self.act_buf = self.act_buf.stack()
            self.rew_buf = self.rew_buf.stack()

            self.V_hats = self.V_hats.stack()
            self.gae = self.gae.stack()

    #@tf.function
    def loss(self):
        logits = model.apply(self.obs_buf)
        action_masks = tf.one_hot(self.act_buf, n_acts)
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits),
                                  axis=1)
        return -tf.reduce_mean(self.gae * log_probs)

@tf.function
def action(obs):
    return tf.squeeze(tf.random.categorical(logits=model.apply(tf.reshape(obs, (1, -1))), num_samples=1),
                      axis=(0, 1))

def run_one_episode(buf):
    obs = env.reset()
    done = False

    for i in range(buf.ptr, buf.size):
        act = action(obs)
        new_obs, rew, done, _ = env.step(act.numpy())

        buf.store(obs, act, rew)
        obs = new_obs

        if done:
            break

    if done:
        buf.finish_path()
    else:
        buf.finish_path(tf.squeeze(value_model.apply(obs.reshape(1, -1))))

def train_one_epoch():

    batch = Buffer(obs_shape, n_acts, gam=γ, lam=λ)

    while batch.ptr < batch.size:
        run_one_episode(batch)

    opt.minimize(batch.loss, var_list=model.trainable_weights)

    #print('batch_obs: {}, batch_V_hats: {}'.format(batch.obs.shape,
    #                                               batch.V_hats.shape))
    value_model.fit(batch.obs_buf.numpy(), batch.V_hats.numpy())

    return batch.loss(), batch.rets, batch.lens

#training loop
for i in range(epochs):
    start_time = time.time()
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    duration = time.time() - start_time
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f \t time: %.3f' %
          (i, batch_loss.numpy(),
           sum(batch_rets)/len(batch_rets),
           sum(batch_lens)/len(batch_lens),
          duration))