from dqn import ConvModel, Model_Cartpole, RelayBuffer, Sars
import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
from random import random
import numpy as np
from utils import FrameStackingAndResizingEnv

env = gym.make('Breakout-v0', render_mode = "human")
env = FrameStackingAndResizingEnv(env, 84, 84, 4)


min_rb_size = 10000
sample_size = 2500
env_steps_before_train = 100
tgt_model_update = 150

eps_max = 1.0
eps_min = 0.01

eps_decay = 0.9999955

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



obs = env.reset()
m = ConvModel(env.observation_space.shape, env.action_space.n).to(device)
m.load_state_dict(torch.load("breakout_youtube.pth"))

rb = RelayBuffer()
steps_since_train = 0
epochs_since_tgt = 0

step_num = -1 * min_rb_size
episode_rewards = []
rolling_reward = 0

tq = tqdm()

try:
    while True:
        tq.update(1)
        eps = eps_decay ** (step_num)

        last_obs = obs
        
        # action = m(torch.Tensor(last_obs).unsqueeze(0).to(device)).max(-1)[-1].item()
        action = env.action_space.sample()
        # action = 1

        obs, reward, done, info = env.step(action)
        rolling_reward += reward


        rb.insert(Sars(last_obs, action, reward, obs, done))

        # print(rolling_reward)


        if done:
            episode_rewards.append(rolling_reward)
            rolling_reward = 0
            # obs = env.reset()[0]
            print(rolling_reward)
            break

        steps_since_train += 1
        step_num += 1



except KeyboardInterrupt:
    pass

env.close()