import torch
from dqn import NeuralNet
import gym
import numpy as np

# model = NeuralNet(100800 , 9)
# model.load_state_dict(torch.load('model_weights.pth'))

# env = gym.make('MsPacman-v0', render_mode = "human")

# state = env.reset()[0]

# for t in range(1000):
#     state = state.reshape(-1)
#     action, log_prob = model.act(state)
#     # print(action, log_prob)
#     state, reward, done, terminate, _ = env.step(action)


#####################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNet(4 , 2).to(device)
model.load_state_dict(torch.load('model_weights_cartpole.pth'))

env = gym.make('CartPole-v1', render_mode = "human")

state = env.reset()[0]
rewards = 0
for t in range(10000):
    # state = state.reshape(-1)
    action, log_prob = model.act(state)
    # print(action, log_prob)
    state, reward, done, terminate, _ = env.step(action)
    rewards += reward

    if done or terminate:
        break

print(rewards)