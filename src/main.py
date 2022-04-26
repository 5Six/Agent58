import json
import torch
import numpy as np, time
import gym
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from collections import deque
from Agent import Agent
from Plot import Plot
from Utils import nparray_to_tensor, concat_tensors
from Utils import load_data


def main() -> None:
    # load in config file
    with open("config.json", "r") as f:
        config = json.load(f)

    GAMMA = config["gamma"]
    EPSILON = config["epsilon_start"]
    EPSILON_FINAL = config["epsilon_final"]
    EPSILON_DECAY = config["epsilon_decay"]
    TOTAL_EPISODE_COUNT = config["total_episode_count"]
    LOSS_FUNCTION = config["loss_function"]
    TARGET_UPDATE = config["target_update"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("ALE/Boxing-ram-v5")

    STATE_SPACE = env.observation_space.shape[0]
    ACTION_SPACE = env.action_space.n

    agent = Agent(
        config=config,
        state_space=STATE_SPACE,
        action_space=ACTION_SPACE,
        device=DEVICE,
    )

    plot = Plot(config)
    latest_scores = deque(maxlen=100)  # store the latest 100 scores
    average_latest_scores = []
    score = average_best_score = 0
    t0 = time.time()
    print(f"Training: {config['method']} DQN {config['custom_name']}")

    for i in range(TOTAL_EPISODE_COUNT):
        done = False
        state_current = env.reset()
        state_current = nparray_to_tensor(state_current, DEVICE)
        state_previous = state_current
        timestep = 0

        while not done:

            timestep += 1

            # epsilon decay
            epsilon = np.interp(i, [0, EPSILON_DECAY], [EPSILON, EPSILON_FINAL])

            state_with_diff = concat_tensors(state_current, state_previous)

            # get action and take step
            action = agent.choose_action(epsilon, state_with_diff)
            next_state, reward, done, _ = env.step(action.item())

            next_state = nparray_to_tensor(next_state, DEVICE)
            score += reward
            reward = torch.tensor([reward], device=DEVICE)
            terminal = torch.tensor([done], device=DEVICE)

            next_state_with_diff = concat_tensors(next_state, state_current)

            # store (s, a, r, s+1, bool) in D
            agent.store_transition(
                (state_with_diff, action, next_state_with_diff, reward, terminal)
            )

            # sample random minibatch of (st,at, r, st+1) from D
            minibatch = agent.sample_experience

            # move to the next state
            state_previous = state_current
            state_current = next_state

            if minibatch is None:
                continue

            # learn from NN
            current_q, expected_q, relavent_q = agent.learn(GAMMA, minibatch)

            # calculate loss
            loss = agent.get_loss(current_q, relavent_q, LOSS_FUNCTION)

            # perform gradient descent
            agent.gradient_decent(loss)

            # if time_step % C == 0: theta2 = theta1
            if timestep % TARGET_UPDATE == 0:
                agent.update_target_network()

            if done:
                latest_scores.append(score)
                average_latest_scores.append(np.mean(latest_scores))
                score = 0

                # plot every 10 episodes
                if i % 10 == 0:
                    print(
                        f"Episode {i}; Epsilon {epsilon:.3f}; Time {time.time()-t0:.2f}; Last 100 avg scores {np.mean(latest_scores):.1f}"
                    )
                    plot.get_plot(average_latest_scores)

                # save weight and plot when agent get a high score
                if np.mean(latest_scores) > average_best_score:
                    agent.save_weights()
                    average_best_score = np.mean(latest_scores)

                    if average_best_score > 99:
                        plot.get_plot(average_latest_scores)
                        print("Training complete.")
                        exit(0)

                break


if __name__ == "__main__":
    main()
