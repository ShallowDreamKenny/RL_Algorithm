"""
# File       : GAIL.py
# Time       ：2023-12-14 0:33
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
from tqdm import tqdm

# from  DQN_Family.Learning_DDQN import DQN
import random

class Discriminator(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, a):
        # cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))

class Qnet(torch.nn.Module):
    def __init__(self, state_size, hidden_dim, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))


class DQN:
    def __init__(self, state_size, hidden_dim, action_size,
                 learning_rate, gamma, epsilon, target_update_interval, device, dqn_type='VanillaDQN'):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_interval = target_update_interval
        self.device = device
        self.qnet = Qnet(state_size, hidden_dim, action_size).to(self.device)
        self.target_qnet = Qnet(state_size, hidden_dim, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=learning_rate)
        self.count = 0
        self.dqn_type = dqn_type


    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.qnet(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.qnet(state).max().item

    def update(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.qnet(state).gather(1, action)
        if self.dqn_type == 'DoubleDQN':
            max_action = self.qnet(next_state).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_qnet(next_state).gather(1, max_action)
        else:
            max_next_q_values = self.target_qnet(next_state).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update_interval == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.count += 1

class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d,device):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)
        expert_actions = torch.tensor(expert_a, dtype=torch.int64).to(device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)
        agent_actions = torch.tensor(agent_a, dtype=torch.int64).to(device)
        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'state': agent_s,
            'action': agent_a,
            'reward': rewards,
            'next_state': next_s,
            'done': dones
        }
        self.agent.update(transition_dict)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 64
    gamma = 0.98
    n_iterations = 2000
    batch_size = 64

    lr_d = 1e-5
    critic_lr = 1e-2
    agent = DQN(state_dim, hidden_dim, action_dim, critic_lr, gamma, 0.01, 10, device, 'DoubleDQN')

    gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d, device)

    n_episode = 10000
    return_list = []
    expert_s, expert_a = np.loadtxt("expert_s.txt", delimiter=' '), np.loadtxt("expert_a.txt", delimiter=' ')

    with tqdm(total=n_episode, desc="进度条") as pbar:
        for i in range(n_episode):
            episode_return = 0
            state = env.reset()
            done = False
            state_list = []
            action_list = []
            next_state_list = []
            done_list = []
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list,
                       next_state_list, done_list)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(return_list)))
    plt.plot(iteration_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GAIL on CartPole-v0')
    plt.show()