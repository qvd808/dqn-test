from dqn import Agent_Cartpole, Model_Cartpole, RelayBuffer, Sars
import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
from random import random
import numpy as np

env = gym.make('CartPole-v1', render_mode = "human")
min_rb_size = 10000
sample_size = 2500
env_steps_before_train = 100
tgt_model_update = 150

eps_max = 1.0
eps_min = 0.01

eps_decay = 0.9999955




obs = env.reset()[0]
m = Model_Cartpole(env.observation_space.shape[0], env.action_space.n)
m.load_state_dict(torch.load("cartpole_youtube1.pth"))

rb = RelayBuffer()
steps_since_train = 0
epochs_since_tgt = 0

step_num = -1 * min_rb_size
episode_rewards = []
rolling_reward = 0

tq = tqdm()

try:
    for i in range(100):
        while True:
            tq.update(1)
            eps = eps_decay ** (step_num)

            last_obs = obs

            action = m(torch.Tensor(last_obs)).max(-1)[-1].item()

            obs, reward, done, terminate, info = env.step(action)
            rolling_reward += reward


            rb.insert(Sars(last_obs, action, reward, obs, done))


            if done or rolling_reward == 500:
                episode_rewards.append(rolling_reward)
                obs = env.reset()[0]
                print(rolling_reward)
                rolling_reward = 0
                break

            steps_since_train += 1
            step_num += 1



except KeyboardInterrupt:
    pass

env.close()

print(f"The average rewards is: {np.mean(episode_rewards)}")