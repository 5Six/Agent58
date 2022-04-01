import gym
import ale_py

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

env = gym.make('ALE/Boxing-ram-v5', render_mode='human')
env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset(return_info=True)
env.close()