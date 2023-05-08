import gym
from dqn import NeuralNet_MsPacman
from collections import deque
import torch
import torch.optim as optim
import numpy as np
import random


env = gym.make('MsPacman-v0')

policy = NeuralNet_MsPacman(100800, 9)
optimizer = optim.Adam(policy.parameters(), lr = 0.01)
n_training_episodes = 1000
max_t = 150
gamma = 0.95
print_every = 100

def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Logic for every episode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]
        # For each episode train the agent in maximum of max_t steps
        for t in range(max_t):
            state = state.reshape(-1)
            action, log_prob = policy.act(state)

            ## Get a random action
            gamma_check_point = random.random()
            if gamma_check_point < gamma ** (i_episode//10):
                action = env.action_space.sample()

            saved_log_probs.append(log_prob)
            state, reward, done, terminate, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t) 
        n_steps = len(rewards) 
       
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]   )    
            
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is 
        # added to the standard deviation of the returns to avoid numerical instabilities        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8: PyTorch prefers gradient descent 
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
    return scores


reinforce(policy=policy, optimizer=optimizer, n_training_episodes=n_training_episodes, max_t=max_t, gamma=1, print_every=100)
torch.save(policy.state_dict(), 'model_weights_mspacman.pth')