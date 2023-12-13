"""
# File       : BehaviorClone.py
# Time       ：2023-12-13 23:29
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from  DQN_Family.Learning_DQN import *
import random

class Qnet(torch.nn.Module):
    def __init__(self, state_size, hidden_dim, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))

class BehaviorClone:
    def __init__(self, state_size, hidden_size, action_size, lr, device, epsilon = 0.01):
        self.policy = Qnet(state_size, hidden_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device
        self.epsilon = epsilon
        self.action_size = action_size

    def learn(self, state, action):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        actions = torch.tensor(action, dtype=int).view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions))
        bc_loss = torch.mean(-log_probs)
        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.policy(state).argmax().item()
        return action

def test_agent(agent, env, n_episode, render=False):
    return_list = []
    for i in range(n_episode):
        i_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            state, reward, done, _ = env.step(action)
            i_return += reward
            if render:
                env.render()
        return_list.append(i_return)
    return np.mean(return_list)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 64
    lr = 1e-3
    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)
    n_iterations = 1000
    batch_size = 64
    test_returns = []

    expert_s, expert_a = np.loadtxt("expert_s.txt", delimiter=' '), np.loadtxt("expert_a.txt", delimiter=' ')

    with tqdm(total=n_iterations, desc='Iter %d' % n_iterations) as pbar:
        for i in range(n_iterations):
            sample_indices = np.random.randint(low=0, high=expert_s.shape[0], size=batch_size)
            bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
            if i > n_iterations - 1:
                current_return = test_agent(bc_agent, env, 5, render = True)
            else:
                current_return = test_agent(bc_agent, env, 5, render = False)
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(test_returns)))
    plt.plot(iteration_list, test_returns)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('BC on {}'.format(env_name))
    plt.show()