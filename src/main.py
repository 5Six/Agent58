import gym
import ale_py
import Replay
import DQN

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

env = gym.make('ALE/Boxing-ram-v5', render_mode='human')
env.reset()

done = 0

flag = 0

while not done:
    action = 1
    state, reward, done, info = env.step(action)
    # print(env.get_keys_to_action())
    print(env.get_action_meanings())
    if flag == 0:
        print(info)
        flag = 1

    if done:
        observation, info = env.reset(return_info=True)
    

env.close()