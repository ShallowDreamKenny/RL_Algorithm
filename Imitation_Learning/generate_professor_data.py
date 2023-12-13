import sys
sys.path.append('../')
"""
# File       : generate_professor_data.py
# Time       ：2023-12-13 21:21
# Author     ：Kust Kenny
# version    ：python 3.8
# Description： use ppo policy to generate the professor data to train the imitation learning model.
"""
import numpy as np
import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import torch.nn as nn
from tqdm import tqdm
import random
import RL_utils

from DQN_Family import Learning_DDQN


def sample_expert_data(n_episode, agent):
    states = []
    actions = []
    for episode in range(n_episode):
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            states.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
    return np.array(states), np.array(actions)


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
    replay_buffer = RL_utils.ReplayBuffer(buffer_size)

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Learning_DDQN.DQN(state_dim, hidden_dim, action_dim, critic_lr, gamma, 0.01, 10, device, 'DoubleDQN')

    return_list = RL_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

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

    expert_s, expert_a = sample_expert_data(1, agent)
    np.savetxt("expert_s.txt", expert_s)
    np.savetxt("expert_a.txt", expert_a)



