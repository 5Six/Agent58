import gym
import ale_py
import Replay
import DQN

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

env = gym.make('ALE/Boxing-ram-v5', render_mode='human')
env.reset()

flag = 0

SEED = 5
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.15
EPISODES = 150
AGENTS = 2
PLANNING_STEPS = 5
MIN_EPSILON = 0.0
MIN_EPISODE_DECAY = 5



TOTAL_EPISODE_COUNT = 100
i = 0

while i < TOTAL_EPISODE_COUNT:
    done = False
    time_step = 1
    #init state S1
    while not done:
        


        time_step += 1 
    i+=1

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