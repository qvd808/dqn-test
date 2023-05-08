import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from random import sample, random
from utils import FrameStackingAndResizingEnv
import gym
import wandb
from tqdm import tqdm
import sys

@dataclass
class Sars:
    state: any
    action: int
    reward: float
    next_state: any
    done: bool



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RelayBuffer:
    def __init__(self, max_size = 100000) -> None:
        self.max_size = max_size
        self.size = 0
        # self.buffer = deque(maxlen=max_size)
        self.buffer = [None] * (max_size + 1)
        self.buffer[0] = Sars(0, 0, -1 * sys.maxsize, 0, True)
        self.FRONT = 1
    
    def parent(self, pos):
        return pos // 2
    
    def leftChild(self, pos):
        return 2 * pos
    
    def rightChild(self, pos):
        return (2 * pos) + 1
    
    def isLeaf(self, pos):
        return pos * 2 > self.size
    
    def swap(self, fpos, spos):
        self.buffer[fpos], self.buffer[spos] = self.buffer[spos], self.buffer[fpos]

    def minHeapify(self, pos):
        # If the node is a non-leaf node and greater
        # than any of its child
        if not self.isLeaf(pos):
            if (self.buffer[pos].reward > self.buffer[self.leftChild(pos)].reward or 
               self.buffer[pos].reward > self.buffer[self.rightChild(pos)].reward):
  
                # Swap with the left child and bufferify
                # the left child
                if self.buffer[self.leftChild(pos)].reward < self.buffer[self.rightChild(pos)].reward:
                    self.swap(pos, self.leftChild(pos))
                    self.minHeapify(self.leftChild(pos))
  
                # Swap with the right child and heapify
                # the right child
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.minHeapify(self.rightChild(pos))    

    # Function to insert a node into the heap
    def insert(self, element):
        if self.size >= self.max_size :
            self.remove()
            self.insert(element)
            return
        self.size+= 1
        self.buffer[self.size] = element
  
        current = self.size
        # import ipdb; ipdb.set_trace()
        while self.buffer[current].reward < self.buffer[self.parent(current)].reward:
            self.swap(current, self.parent(current))
            current = self.parent(current)

    # Function to print the contents of the buffer
    def print(self):
        for i in range(1, (self.size//2)+1):
            print(" PARENT : "+ str(self.buffer[i])+" LEFT CHILD : "+ 
                                str(self.buffer[2 * i])+" RIGHT CHILD : "+
                                str(self.buffer[2 * i + 1]))
  
  
    # Function to build the min heap using
    # the minHeapify function
    def minHeap(self):
  
        for pos in range(self.size//2, 0, -1):
            self.minHeapify(pos)
  
    # Function to remove and return the minimum
    # element from the heap
    def remove(self):
  
        popped = self.buffer[self.FRONT]
        self.buffer[self.FRONT] = self.buffer[self.size]
        self.size-= 1
        self.minHeapify(self.FRONT)
        return popped

    # def insert(self, sars):
    #     self.buffer[self.idx % self.buffer_size] = sars
    #     self.idx += 1
    #     # self.buffer.append(sars)
    
    def sample(self, num_samples):
        if self.size < self.max_size:
            return sample(self.buffer[1:self.size], num_samples)
        return sample(self.buffer, num_samples)
    
class ConvModel(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.0001) -> None:
        super(ConvModel, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, (8, 8), stride=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (4, 4), stride=(2, 2)),
            torch.nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape))
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
        
        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(fc_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions),
        )

        self.optim = optim.Adam(self.parameters(), lr = lr)

    def forward(self, x):
        conv_latent = self.conv_net(x/255.0)
        return self.fc_net(conv_latent.view((conv_latent.shape[0], -1)))
    



env = gym.make('Breakout-v0')
env = FrameStackingAndResizingEnv(env, 84, 84, 4)

test_env = gym.make('Breakout-v0')
test_env = FrameStackingAndResizingEnv(test_env, 84, 84, 4)

memory_size = 1000000
min_rb_size = 70000
sample_size = 4000
env_steps_before_train = 200
tgt_model_update = 750
wandb.init(project="dqn-tutorial", name="dqn-breakout")

eps_max = 1.0
eps_min = 0.01

eps_decay = 0.99999885

def run_test_episode(model, env, device,  max_steps = 1000): # -> rewards, movie
    frames = []
    obs = env.reset()
    frames.append(env.frame)
    
    idx = 0
    done = False
    rewards = 0

    while not done and idx < max_steps:
        action = model(torch.Tensor(last_obs).unsqueeze(0).to(device)).max(-1)[-1].item()
        obs, reward, done, info = env.step(action)
        rewards += reward
        frames.append(env.frame)
    
    return rewards, np.stack(frames, 0)



def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transition, tgt, num_actions, device):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transition])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transition])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transition])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transition])).to(device)
    actions = [s.action for s in state_transition]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]
    
    model.optim.zero_grad()
    qvals = model(cur_states)
    # import ipdb; ipdb.set_trace()

    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)


    loss = ((rewards +  0.994 * mask[:, 0] * qvals_next - torch.sum(qvals * one_hot_actions, -1)) ** 2).mean()
    # loss_fn = torch.nn.SmoothL1Loss()
    # loss = loss_fn(torch.sum(qvals * one_hot_actions), rewards + mask[:, 0] * qvals_next * 0.98 )

    loss.backward()
    model.optim.step()
    

    return loss


obs = env.reset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

m = ConvModel(env.observation_space.shape, env.action_space.n).to(device)
tgt = ConvModel(env.observation_space.shape, env.action_space.n).to(device)

update_tgt_model(m, tgt)

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

        if random() < eps:
            action = env.action_space.sample()
        else:
            # import ipdb; ipdb.set_trace()
            action = m(torch.Tensor(last_obs).unsqueeze(0).to(device)).max(-1)[-1].item()

        obs, reward, done, info = env.step(action)
        rolling_reward += reward

        reward = reward / 10.0

        rb.insert(Sars(last_obs, action, reward, obs, done))



        if done or (step_num % 500 == 0 and step_num > 0):
            episode_rewards.append(rolling_reward)
            rolling_reward = 0
            obs = env.reset()

        steps_since_train += 1
        step_num += 1

        if rb.size > min_rb_size and steps_since_train > env_steps_before_train:
            epochs_since_tgt += 1
            
            loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n, device)
            wandb.log({'loss': loss.detach().cpu().item(), 'eps': eps, 'avg_reward': np.mean(episode_rewards)}, step = step_num)
            # print(step_num, loss.detach().item())
            episode_rewards = []
            if epochs_since_tgt > tgt_model_update:
                print("Updating target model")
                update_tgt_model(m, tgt)
                torch.save(m.state_dict(), f"breakout_{step_num}.pth")

                epochs_since_tgt = 0
                
            steps_since_train = 0

except KeyboardInterrupt:
    torch.save(m.state_dict(), "breakout_youtube.pth")
    pass

env.close()
