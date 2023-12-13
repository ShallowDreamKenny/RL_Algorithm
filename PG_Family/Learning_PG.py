"""
# File       : Learning_PG.py
# Time       ：2023-12-12 9:24
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import sys
sys.path.append('../')

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import RL_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_size, hidden_dim, action_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class REINFORCE:
    def __init__(self, state_size, hidden_dim, action_size, learning_rate, gamma, device):
        self.policy = PolicyNet(state_size, hidden_dim, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        rewards = transition_dict['rewards']
        states = transition_dict['states']
        actions = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            state = torch.tensor([states[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([actions[i]]).view(-1, 1).to(self.device)
            log_probs = torch.log(self.policy(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_probs * G
            loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    learning_rate = 0.001
    gamma = 0.98
    hidden_dim = 128
    num_episodes = 1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v0')
    env.seed(seed=0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

    return_list = []

    for i in range(10):
        with tqdm(total= int(num_episodes/10), desc='Iteration %d' % i, colour='green') as pbar:
            for episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'next_states': [],
                    'dones': []
                }
                state = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)

                if(episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)


    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format('CartPole-v0'))
    plt.show()

    mv_return = RL_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format('CartPole-v0'))
    plt.show()

