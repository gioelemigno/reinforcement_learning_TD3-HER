#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ReplayBuffer
import TD3
import gym 
import json
import Normalizer
import numpy as np
import tensorflow as tf
import keras
import os

print("Tensorflow version %s" %tf.__version__)

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[2]:
REF_EXPERIMENT = '100_epochs_RANDOM_START_new_seed'#'100_epochs_RANDOM_START'#'50_epochs_RANDOM_START'#'50_epochs'


# In[3]:
ROOT = '.'
FOLDER_BEST_WEIGHTS = os.path.join(ROOT, REF_EXPERIMENT, 'best_weights')
FOLDER_EXPORT_RESULTS = os.path.join(ROOT, REF_EXPERIMENT, 'results')
FOLDER_BEST_NORMALIZERS = os.path.join(ROOT, REF_EXPERIMENT, 'best_normalizers')


# In[4]:
ENV_NAME = 'FetchPickAndPlace-v1'
RND_SEED = 20210122
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5

DISCOUNT = 0.99
TAU = 0.005
POLICY_FREQ = 2


# In[5]:
#set random seed 
tf.random.set_seed(RND_SEED)
np.random.seed(RND_SEED)

env = gym.make(ENV_NAME)
env.seed(RND_SEED)


# In[6]:
observation_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]

state_dim = observation_dim + goal_dim
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


# In[7]:
CLIP_RANGE = 5 
obs_normalizer = Normalizer.normalizer(size=observation_dim, default_clip_range=CLIP_RANGE)
goal_normalizer = Normalizer.normalizer(size=goal_dim, default_clip_range=CLIP_RANGE)


# In[8]:
def choose_action(state, noise=True):
    action_policy = policy.pi(np.array(state))
    if noise:
        noise = np.random.normal(0,  max_value_action * EXPLORATION_NOISE , size=action_dim)
        action = action_policy + noise 
        action = tf.clip_by_value(action, clip_value_min=-max_value_action, clip_value_max=max_value_action)
        action = tf.reshape(action, [action_dim])
        action = action.numpy()
    else:
        action = tf.reshape(action_policy, [action_dim])
    return action

def split_obs(obs_dict):
    obs = obs_dict['observation']
    achieved_goal = obs_dict['achieved_goal']
    goal = obs_dict['desired_goal']
    return obs, achieved_goal, goal

def build_state(obs, goal):
    return np.concatenate((obs, goal))

def split_state(state):
    if len(state) != observation_dim + goal_dim:
        print("Error")
        return None
    obs = state[:observation_dim]
    goal = state[observation_dim:]
    
    return obs, goal

def normalize(obs, goal):
    obs_norm = obs_normalizer.normalize(obs)
    goal_norm = goal_normalizer.normalize(goal)
    return obs_norm, goal_norm

def preprocess_state(state):
    obs, goal = split_state(state)
    obs_norm, goal_norm = normalize(obs, goal)
    state_norm = build_state(obs_norm, goal_norm)
    #print("norm=" + str(state_norm))
    return state_norm


# In[11]:
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    total_success_rate = []
    for _ in range(eval_episodes):
        per_success_rate = []
        obs_dict, done = eval_env.reset(), False
        obs, achieved_goal, g = split_obs(obs_dict)
        for _ in range(eval_env._max_episode_steps):
            obs_norm, goal_norm = normalize(obs, g)
            state_norm = build_state(obs_norm, goal_norm)
            action = choose_action(state_norm, noise=False)

            observation_new, r, done, info = eval_env.step(action)
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
            
            if done:
                break
        total_success_rate.append(per_success_rate)

    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate)

    print("--------------------------------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {local_success_rate*100:.2f}% success rate")
    print("--------------------------------------------------------------")
    return local_success_rate


# In[12]:
# evaluation policy without training
eval_policy(policy, ENV_NAME, RND_SEED)


# In[13]:
import pickle
# LOAD model and normalizers
policy.load_weights_networks(FOLDER_BEST_WEIGHTS)

filename = os.path.join(FOLDER_BEST_NORMALIZERS, "obs_normalizer")
infile = open(filename,'rb')
obs_normalizer = pickle.load(infile)
infile.close()

filename = os.path.join(FOLDER_BEST_NORMALIZERS, "goal_normalizer")
infile = open(filename,'rb')
goal_normalizer = pickle.load(infile)
infile.close()


# In[14]:
# evaluation policy AFTER training
eval_policy(policy, ENV_NAME, RND_SEED)


# In[15]:
#plot
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
    anim.save(os.path.join(path, filename), writer='imagemagick', fps=25)#60)

def plot_experiment_policy(policy, env_name, seed, output_gif_name):
    frames = []
    env_plot = gym.make(ENV_NAME)
    env_plot.seed(seed)

    max_t = 100
    t=0
    obs_dict, done = env_plot.reset(), False
    obs, achieved_goal, g = split_obs(obs_dict)

    while not done and t<max_t:
        frames.append(env_plot.render(mode="rgb_array"))
        obs_norm, goal_norm = normalize(obs, g)
        state_norm = build_state(obs_norm, goal_norm)
        action = choose_action(state_norm, noise=False)
        observation_new, r, done, info = env_plot.step(action)
        obs = observation_new['observation']
        g = observation_new['desired_goal']
        #if info['is_success'] == 1:
        #    break
        t += 1

    env_plot.close()
    print("Exporting ...")
    save_frames_as_gif(frames, path=FOLDER_EXPORT_RESULTS, filename=output_gif_name)

NUMBER_TEST = 20
for i in range(NUMBER_TEST):
    seed = RND_SEED + i*100
    plot_experiment_policy(policy, ENV_NAME, seed, REF_EXPERIMENT + "_test_" + str(seed) +".gif")






