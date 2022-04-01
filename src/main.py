import gym
import ale_py

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

env = gym.make('ALE/Boxing-ram-v5')