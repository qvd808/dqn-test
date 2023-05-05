from dqn import Agent_Cartpole, Model_Cartpole, RelayBuffer, Sars
import gym
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from random import random
import numpy as np

env = gym.make('CartPole-v1')
min_rb_size = 10000
sample_size = 2500
env_steps_before_train = 100
tgt_model_update = 150
wandb.init(project="dqn-tutorial", name="dqn-cartpole")

eps_max = 1.0
eps_min = 0.01

eps_decay = 0.99998

def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transition, tgt, num_actions):
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transition]))
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transition]))
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transition]))
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transition]))
    actions = [s.action for s in state_transition]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]
    
    model.optim.zero_grad()
    qvals = model(cur_states)
    # import ipdb; ipdb.set_trace()

    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)


    loss = ((rewards +  mask[:, 0] * qvals_next - torch.sum(qvals * one_hot_actions, -1)) ** 2).mean()
    loss.backward()
    model.optim.step()
    

    return loss


obs = env.reset()[0]
m = Model_Cartpole(env.observation_space.shape[0], env.action_space.n)
tgt = Model_Cartpole(env.observation_space.shape[0], env.action_space.n)

update_tgt_model(m, tgt)

qvals = m(torch.tensor(obs))
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
            action = m(torch.Tensor(last_obs)).max(-1)[-1].item()

        obs, reward, done, terminate, info = env.step(action)
        rolling_reward += reward

        reward = reward / 100.0

        rb.insert(Sars(last_obs, action, reward, obs, done))



        if done:
            episode_rewards.append(rolling_reward)
            rolling_reward = 0
            obs = env.reset()[0]

        steps_since_train += 1
        step_num += 1

        if len(rb.buffer) > min_rb_size and steps_since_train > env_steps_before_train:
            epochs_since_tgt += 1
            
            loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n)
            wandb.log({'loss': loss.detach().item(), 'eps': eps, 'avg_reward': np.mean(episode_rewards)}, step = step_num)
            # print(step_num, loss.detach().item())
            episode_rewards = []
            if epochs_since_tgt > tgt_model_update:
                print("Updating target model")
                update_tgt_model(m, tgt)
                epochs_since_tgt = 0
                
            steps_since_train = 0

except KeyboardInterrupt:
    torch.save(m.state_dict(), "cartpole_youtube.pth")
    pass

env.close()