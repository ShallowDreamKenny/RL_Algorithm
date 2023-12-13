"""
# File       : Learning_SARSA.py
# Time       ：2023-12-08 22:30
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import time
import gym
import numpy as np

env = gym.make('FrozenLake-v1')
render = True
running_reward = None

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = .85     #学习率α
lambd = .99  #折扣率
num_episodes = 10000 # 迭代次数
rList = []   #记录每次迭代的奖励值，用于观测智能体是否有进步

for i in range(num_episodes):
    episode_time = time.time()
    state = env.reset()
    rAll = 0

    for t in range(100):
        if render and i > num_episodes - 5: env.render()

        a = a_next if t != 0 else np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i+1)))
        next_state, reward, done, info = env.step(a)
        a_next = np.argmax(Q[next_state, :] + np.random.randn(1, env.action_space.n) * (1. / (i+1)))

        Q[state, a] = Q[state, a] + lr * (reward + lambd * Q[next_state, a_next] - Q[state, a])
        rAll += reward
        state = next_state
        if done == True:
            break

    rList.append(rAll)
    running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
    print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
          (i, num_episodes, rAll, running_reward, time.time() - episode_time))

print("Final Q-Table Values:/n %s" % Q)