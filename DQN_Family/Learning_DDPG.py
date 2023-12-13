"""
# File       : Learning_DDPG.py
# Time       ：2023-12-12 11:25
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import sys
sys.path.append('../')

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import random
import matplotlib.pyplot as plt
import RL_utils

class PolicyNet(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.action_bound = action_bound # action_bound 是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x)) * self.action_bound
        return x

class QNet(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDPG:
    def __init__(self, state_size, hidden_size, action_size, action_bound,
                 sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_size, hidden_size, action_size, action_bound).to(device)
        self.actor_target = PolicyNet(state_size, hidden_size,action_size, action_bound).to(device)
        self.critic = QNet(state_size, hidden_size, action_size).to(device)
        self.critic_target = QNet(state_size, hidden_size, action_size).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.sigma = sigma

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        action = action + self.sigma * np.random.randn(self.action_size)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.critic_target(next_states, self.actor_target(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1-dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)


if __name__ == '__main__':
    actor_lr = 3e-4
    critic_lr = 3e-3
    num_episodes = 500
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01  # 高斯噪声标准差
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = RL_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

    return_list = RL_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.show()

    mv_return = RL_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.show()