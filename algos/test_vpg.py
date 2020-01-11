import tensorflow as tf
import gym
import pybullet_envs

import time

from baselines import logger

env = gym.make('Walker2DBulletEnv-v0')
env.render(mode='human')
obs_shape = env.observation_space.shape
n_acts = env.action_space.shape[0]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='tanh', input_shape=obs_shape),
    tf.keras.layers.Dense(n_acts)
])
model.summary()

load_path = './model'
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
ckpt.restore(manager.latest_checkpoint)
print("Restoring from {}".format(manager.latest_checkpoint))



def action(obs):
    return tf.squeeze(model(tf.expand_dims(obs, 0)), axis=0)

obs = env.reset()
while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        act = action(obs)
        obs, rew, done, _ = env.step(act.numpy())
        episode_rew += rew
    print("Episode reward", episode_rew)
