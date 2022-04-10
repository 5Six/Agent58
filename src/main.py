import gym
import ale_py
import torch
from collections import deque

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

env = gym.make('Boxing-ram-v4', render_mode='human')
env.reset()

capacity = 1000
replay = deque([], capacity)

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset(return_info=True)
    

env.close()