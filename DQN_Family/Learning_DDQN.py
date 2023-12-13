import sys
sys.path.append('../')

import collections
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import RL_utils


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    def __init__(self, state_size, hidden_dim, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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


if __name__ == '__main__':
    render = False

    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    batch_size = 64
    epsilon = 0.01
    target_update_interval = 10
    buffer_size = 10000
    minimal_size = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(seed=0)
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update_interval, device, 'DoubleDQN')
    # return_list = RL_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if i == 9 and episode > int(num_episodes / 10) - 10:
                        env.render()
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'state': b_s,
                            'action': b_a,
                            'reward': b_r,
                            'next_state': b_ns,
                            'done': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = RL_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()
