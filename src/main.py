import json
from typing import Final
import torch
import numpy as np
import gym

from Agent import Agent
from Utils import nparray_to_tensor
from Utils import load_data


def main() -> None:
    # load in config file
    with open("./config.json", "r") as f:
        config = json.load(f)

    GAMMA: Final = config['gamma']
    EPSILON: Final = config['epsilon_start']
    EPSILON_FINAL: Final = config['epsilon_final']
    EPSILON_DECAY: Final = config['epsilon_decay']
    TOTAL_EPISODE_COUNT: Final = config['total_episode_count']
    LOSS_FUNCTION: Final = config['loss_function']
    TARGET_UPDATE: Final = config['target_update']

    DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("ALE/Boxing-ram-v5")

    STATE_SPACE: Final = env.observation_space.shape[0]
    ACTION_SPACE: Final = env.action_space.n

    agent = Agent(
        config=config,
        state_space=STATE_SPACE,
        action_space=ACTION_SPACE,
        device=DEVICE,
    )

 

  
    for i in range(TOTAL_EPISODE_COUNT):
        done = False
        state_current = env.reset()
        state_current = nparray_to_tensor(state_current, DEVICE)
        state_previous = state_current
        #state = nparray_to_tensor(state, DEVICE)

        while not done:

            
            state_with_diff = torch.cat((state_current[0], state_current[0] - state_previous[0]))
          
            # epsilon decay
            epsilon = np.interp(i, [0, EPSILON_DECAY], [EPSILON, EPSILON_FINAL])
            action = agent.choose_action(epsilon, state_with_diff)
            next_state, reward, done, _ = env.step(action.item())
            next_state = nparray_to_tensor(next_state, DEVICE)
            reward = torch.tensor([reward], device=DEVICE)
            terminal = torch.tensor([done], device=DEVICE)
            
            next_state_with_diff = torch.cat((next_state[0], next_state[0] - state_with_diff[0]))
            
            state_with_diff = state_with_diff[None, :]
            next_state_with_diff = next_state_with_diff[None, :]

            # store (s, a, r, s+1, bool) in D
            agent.store_transition((state_with_diff, action, next_state_with_diff, reward, terminal))

            # sample random minibatch of (st,at, r, st+1) from D
            minibatch = agent.sample_experience

            if minibatch is None:
                state_previous = state_current
                state_current = next_state
                continue

            # learn from NN
            current_q, expected_q, relavent_q = agent.learn(GAMMA, minibatch)

            # calculate loss
            loss = agent.get_loss(current_q, relavent_q, LOSS_FUNCTION)

            # perform gradient descent
            agent.gradient_decent(loss)

            # if time_step % C == 0: theta2 = theta1
            if i % TARGET_UPDATE:
                agent.update_target_network()

            # move to the next state
            state_previous = state_current
            state_current = next_state
            
            
            if done:
                state_current = env.reset()
                state_previous = state_current
                break


if __name__ == "__main__":
    main()

