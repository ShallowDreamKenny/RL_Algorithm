"""
# File       : DQN_for_Breakout.py
# Time       ：2023-12-10 22:19
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import sys
sys.path.append('../')

import cv2
import gym
import numpy as np

env = gym.make('Breakout-v0', render_mode='human')
_ = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        state = env.reset()
    # env.render()

def preprocess(img):
    img_temp = img[31:195]
    img_temp = img_temp.mean(axis = 2)
    img_temp = cv2.resize(img_temp, ())
