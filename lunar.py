import gym
from dqn import Agent
import numpy as np

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=8, lr=0.003)
    scores, eps_his = [], []

    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()[0]

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, terminate, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, done)

            agent.learn()
            obs = obs_

        scores.append(score)
        eps_his.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print(f'episode {i} scores {score} average score {avg_score}, epsilon {agent.epsilon}')
    
    agent.save()