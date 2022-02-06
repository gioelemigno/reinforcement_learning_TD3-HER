#!/usr/bin/env python
# coding: utf-8

# In[8]:


import ReplayBuffer
import TD3
import gym 

import numpy as np
import tensorflow as tf
import keras
import os
print("Tensorflow version %s" %tf.__version__)

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[16]:
REF_EXPERIMENT = 'Test_200k'


# In[17]:
ROOT = '.'

FOLDER_EXPORT_RESULTS = os.path.join(ROOT, REF_EXPERIMENT, 'results')
FOLDER_IMPORT_WEIGHTS = os.path.join(ROOT, REF_EXPERIMENT, 'best_weights')


# In[11]:
ENV_NAME = 'HalfCheetah-v2'
RND_SEED = 20210122
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
BATCH_SIZE = 256
DISCOUNT = 0.99
TAU = 0.005
POLICY_FREQ = 2


# In[12]:
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.pi(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


# In[13]:
#set random seed 
tf.random.set_seed(RND_SEED)
np.random.seed(RND_SEED)

env = gym.make(ENV_NAME)
env.seed(RND_SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_value_action = float(env.action_space.high[0])


kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_value_action": max_value_action,
    "discount": DISCOUNT,
    "tau": TAU,
    "policy_noise": POLICY_NOISE* max_value_action,
    "noise_clip": NOISE_CLIP * max_value_action,
    "policy_freq": POLICY_FREQ,
}

policy = TD3.TD3(**kwargs)
replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim)


# In[14]:
eval_policy(policy, ENV_NAME, RND_SEED)


# In[18]:
policy.load_weights_networks(FOLDER_IMPORT_WEIGHTS)


# In[19]:
eval_policy(policy, ENV_NAME, RND_SEED)


# In[ ]:
# source: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
# convert GIF to video 
# ffmpeg -f gif -i ./gym_animation.gif out.mp4
# NOTE:
# run in terminal, no conda since ffmpeg doesn't work with it 
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import animation
import matplotlib.pyplot as plt
import gym 

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=25)#60)

#Make gym env
env = gym.make(ENV_NAME)
frames = []
env.seed(RND_SEED + 100)
state, done = env.reset(), False
max_t = 100
t=0
while not done and t<max_t:
    frames.append(env.render(mode="rgb_array"))
    action = policy.pi(np.array(state))
    state, reward, done, _ = env.step(action)
    t += 1

env.close()
print("Exporting ...")
save_frames_as_gif(frames)

