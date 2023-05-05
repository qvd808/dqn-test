import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from random import sample
from collections import deque

@dataclass
class Sars:
    state: any
    action: int
    reward: float
    next_state: any
    done: bool



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(device)
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        # self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
class NeuralNet_MsPacman(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(NeuralNet_MsPacman, self).__init__()
        self.fc1 = nn.Linear(input_size,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)
    
    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

    

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-4) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions = n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_mem = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.actions_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.reward_mem[index] = reward
        self.actions_mem[index] = action
        self.terminal_mem[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action  = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def choose_action_final(self, observation):
        state = torch.tensor([observation]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action  = torch.argmax(actions).item()    
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        else:
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            state_batch = torch.tensor(self.state_mem[batch]).to(self.Q_eval.device)
            new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.Q_eval.device)
            reward_batch = torch.tensor(self.reward_mem[batch]).to(self.Q_eval.device)
            terminal_batch = torch.tensor(self.terminal_mem[batch]).to(self.Q_eval.device)

            actions_batch = torch.tensor(self.actions_mem[batch])

            # import ipdb; ipdb.set_trace()

            q_eval = self.Q_eval.forward(state_batch)[torch.tensor(batch_index).long(), torch.tensor(actions_batch).long()]
            q_next = self.Q_eval.forward(new_state_batch)

            q_next[terminal_batch] = 0
            q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def save(self):
        torch.save(self.Q_eval.state_dict(), "lunar-lander.pth")


class RelayBuffer:
    def __init__(self, buffer_size = 100000) -> None:
        self.buffer_size = buffer_size
        # self.buffer = deque(maxlen=buffer_size)
        self.buffer = [None] * buffer_size
        self.idx = 0

    def insert(self, sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1
        # self.buffer.append(sars)
    
    def sample(self, num_samples):
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], num_samples)
        return sample(self.buffer, num_samples)


class Model_Cartpole(nn.Module):
    def __init__(self, obs_shape, n_actions) -> None:
        super(Model_Cartpole, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions)
        )

        self.optim = optim.Adam(self.parameters(),lr=0.0001)

    def forward(self, x):
        return self.net(x)
    

class Agent_Cartpole:
    def __init__(self, model) -> None:
        self.model = model
    
    def get_actions(self, obs):
        #obs shape is (N, 4)
        q_val = self.model(obs)

        # q_vals is (N, 2)

        return q_val.max(-1)[0]
    
