import gym
from dqn import NeuralNet
from collections import deque
import torch
import torch.optim as optim
import numpy as np


env = gym.make('CartPole-v1', render_mode = "human")

# net = NeuralNet(4 , 2)

# scores_deque = deque(maxlen=100)
# scores = []

# max_step_in_eps = 800
# gamma = 1
# optimizer = optim.Adam(net.parameters(), lr = 0.01)
# epsilon = 0.9

# for i_episode in range(1, 2000+1):
#     saved_log_probs = []
#     rewards = []
#     state = env.reset()[0]

#     for t in range(max_step_in_eps):
#         # state = state.reshape(-1)
#         # print(state)
#         action, log_prob = net.act(state)
#         # print(action, log_prob)
#         saved_log_probs.append(log_prob)
#         state, reward, done, terminate, _ = env.step(action)
#         rewards.append(reward)
#         if done or terminate:
#             break
    
#     scores_deque.append(sum(rewards))
#     scores.append(sum(rewards))

#     returns = deque(maxlen=max_step_in_eps)
#     n_steps = len(rewards)

#     for t in range(n_steps)[::-1]:
#         disc_return_t = (returns[0] if len(returns) > 0 else 0)
#         returns.appendleft(gamma * disc_return_t + rewards[t])

#     eps = np.finfo(np.float32).eps.item()
#     returns = torch.tensor(returns)
#     returns = (returns - returns.mean()) / (returns.std() + eps)

#     policy_loss = []
#     for log_prob, disc_return in zip(saved_log_probs, returns):
#         policy_loss.append(-log_prob * disc_return)
#     policy_loss = torch.cat(policy_loss).sum()
    
#     # Line 8: PyTorch prefers gradient descent 
#     optimizer.zero_grad()
#     policy_loss.backward()
#     optimizer.step()
    
#     if i_episode % 100 == 0:
#         print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))


# torch.save(net.state_dict(), 'model_weights_cartpole.pth')


policy = NeuralNet(4, 2)
optimizer = optim.Adam(policy.parameters(), lr = 0.01)
n_training_episodes = 1000
max_t = 1000
gamma = 1
print_every = 100

def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
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
        # Compute the discounted returns at each timestep,
        # as 
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity 
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...
        
        # Given this formulation, the returns at each timestep t can be computed 
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order 
        # to avoid computing them multiple times)
        
        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...
        
        
        ## Given the above, we calculate the returns at timestep t as: 
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed 
        ## if we were to do it from first to last.
        
        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
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
torch.save(policy.state_dict(), 'model_weights_cartpole.pth')