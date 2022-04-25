import json
from typing import Final
import torch
import numpy as np
import gym
from collections import deque

from Agent import Agent
from Utils import nparray_to_tensor
from Utils import load_data
from Utils import Plot


def main() -> None:
    METHOD: Final = "double"
    CUSTOM_NAME: Final = "test"

    # load in config file
    with open("config.json", "r") as f:
        config = json.load(f)

    GAMMA: Final = config['gamma']
    EPSILON: Final = config['epsilon_start']
    EPSILON_FINAL: Final = config['epsilon_final']
    EPSILON_DECAY: Final = config['epsilon_decay']
    TOTAL_EPISODE_COUNT: Final = config['total_episode_count']
    LOSS_FUNCTION: Final = config['loss_function']
    TARGET_UPDATE: Final = config['target_update']
    NET_SAVE_PATH: Final = f"net/net_boxing-v5_{METHOD}DQN_{CUSTOM_NAME}.pth"

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

    plot = Plot(METHOD, CUSTOM_NAME)
    latest_scores = deque(maxlen=100) # store the latest 100 scores
    average_latest_scores = []
    score = average_best_score = 0

    for i in range(TOTAL_EPISODE_COUNT):
        done = False
        state_current = env.reset()
        state_current = nparray_to_tensor(state_current, DEVICE)
        state_previous = state_current
        #state = nparray_to_tensor(state, DEVICE)
        timestep = 0

        while not done:
            timestep += 1
            state_with_diff = torch.cat((state_current[0], state_current[0] - state_previous[0]))
          
            # epsilon decay
            epsilon = np.interp(i, [0, EPSILON_DECAY], [EPSILON, EPSILON_FINAL])
            action = agent.choose_action(epsilon, state_with_diff)
            next_state, reward, done, _ = env.step(action.item())
            next_state = nparray_to_tensor(next_state, DEVICE)
            score += reward
            reward = torch.tensor([reward], device=DEVICE)
            terminal = torch.tensor([done], device=DEVICE)
            
            next_state_with_diff = torch.cat((next_state[0], next_state[0] - state_current[0]))
            
            state_with_diff = state_with_diff[None, :]
            next_state_with_diff = next_state_with_diff[None, :]

            #print(next_state_with_diff)
            # store (s, a, r, s+1, bool) in D
            agent.store_transition((state_with_diff, action, next_state_with_diff, reward, terminal))

            # sample random minibatch of (st,at, r, st+1) from D
            minibatch = agent.sample_experience

            state_previous = state_current
            state_current = next_state

            if minibatch is None:              
                continue

            # learn from NN
            current_q, expected_q, relavent_q = agent.learn(GAMMA, minibatch, METHOD)

            # calculate loss
            loss = agent.get_loss(current_q, relavent_q, LOSS_FUNCTION)
            # print(i, ": ", loss)

            # perform gradient descent
            agent.gradient_decent(loss)
            
            #if time_step % C == 0: theta2 = theta1
            if timestep % TARGET_UPDATE == 0:
                agent.update_target_network()

            # move to the next state
            if done:
                latest_scores.append(score)
                average_latest_scores.append(np.mean(latest_scores))
                score = 0

                # plot every 10 episodes
                if i % 10 == 0 and i > 0:
                    plot.get_plot(average_latest_scores)
                    # agent.get_weights(NET_SAVE_PATH)

                # save weight and plot when agent get a high score
                if np.mean(latest_scores) > average_best_score:
                    agent.get_weights(NET_SAVE_PATH)
                    average_best_score = np.mean(latest_scores)

                    if average_best_score > 99:
                        plot.get_plot(average_latest_scores)
                        exit(0)

                break


if __name__ == "__main__":
    main()
