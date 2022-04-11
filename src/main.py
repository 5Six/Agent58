import gym
import ale_py
import Replay
import Net
import Agent

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

env = gym.make('ALE/Boxing-ram-v5', render_mode='human')
env.reset()

flag = 0

TOTAL_EPISODE_COUNT = 100
BATCH_SIZE = 20
C = 10

epsilon, alpha, gamma = 0.1, 0.9, 0.1

current_agent = Agent(epsilon, gamma, alpha)
for episode in range(TOTAL_EPISODE_COUNT):
    done = False
    time_step = 1
    state = env.reset()
    while not done:
        action = current_agent.choose_action()
        next_state, reward, done, info = env.step(action)
        
        if done:
            next_state = "Terminal"

        current_agent.buffer.push(state, action, next_state, reward)

        # store (st,at, r, st+1) in D
        # memory.push(state, action, next_state, reward)

        
        # sample random minibatch of (st,at, r, st+1) from D
        minibatch = current_agent.buffer.sample(BATCH_SIZE)
        for i in minibatch:
            yj = 0
            if i.next_state == "Terminal":
                yj = i.reward
            else:
                yj = i.reward + current_agent.GAMMA * current_agent.get_best_arg() 
                
            # perform gradient descent 

        # if time_step % C == 0: theta2 = theta1
        state = next_state
        time_step += 1 


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

# if __name__ == '__main__':
#     main()