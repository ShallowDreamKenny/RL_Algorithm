"""
# File       : RL_utils.py
# Time       ：2023-12-10 10:10
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""

from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import gym

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

def moving_average(a, window):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window:] - cumulative_sum[:-window]) / window
    r = np.arange(1, window-1, 2)
    begin = np.cumsum(a[:window-1])[::2] / r
    end = (np.cumsum(a[:-window:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle,end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc="Iteration %d" % i) as pbar:
            for episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'state': [], 'action': [], 'reward': [], 'done' : [], 'next_state': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['state'].append(state)
                    transition_dict['action'].append(action)
                    transition_dict['reward'].append(reward)
                    transition_dict['next_state'].append(next_state)
                    transition_dict['done'].append(done)
                    state = next_state
                    episode_return += reward
                    if i == 9 and episode > int(num_episodes / 10) - 10:
                        env.render()
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc="Iteration %d" % i) as pbar:
            for episode in range(int(num_episodes/10)):
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
                        b_s, b_a, b_r, b_ns,b_d = replay_buffer.sample(batch_size)
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
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 *i + episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)