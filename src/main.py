import json
from typing import Final
import torch
import numpy as np
import gym

from Agent import Agent
from Utils import nparray_to_tensor
from Utils import load_data


def main() -> None:

    ALPHA: Final = 0.001
    GAMMA: Final = 0.9
    EPSILON: Final = 1.0
    EPSILON_FINAL: Final = 0.05
    EPSILON_DECAY: Final = 1000
    TOTAL_EPISODE_COUNT: Final = 10
    BATCH_SIZE: Final = 5
    MEMORY_CAPACITY: Final = 1000
    LOSS_FUNCTION: Final = "HuberLoss"
    GRADIENT_ALGORITHM: Final = "RMSprop"
    TARGET_UPDATE: Final = 10
    SEED: Final = 10

    # # load in config file
    # with json.open("TO DO"):
    #     pass

    DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("ALE/Boxing-ram-v5")

    STATE_SPACE: Final = env.observation_space.shape[0]
    ACTION_SPACE: Final = env.action_space.n

    agent = Agent(
        learning_rate=ALPHA,
        state_space=STATE_SPACE,
        action_space=ACTION_SPACE,
        gradient_algo=GRADIENT_ALGORITHM,
        device=DEVICE,
        seed=SEED,
        memory_capacity=MEMORY_CAPACITY,
        batch_size=BATCH_SIZE,
    )

    rewards = 0

    for i in range(TOTAL_EPISODE_COUNT):
        done = False
        state = env.reset()
        state = nparray_to_tensor(state, DEVICE)

        while not done:

            # epsilon decay
            epsilon = np.interp(i, [0, EPSILON_DECAY], [EPSILON, EPSILON_FINAL])

            action = agent.choose_action(epsilon, state)
            next_state, reward, done, _ = env.step(action.item())
            next_state = nparray_to_tensor(next_state, DEVICE)
            reward = torch.tensor([reward], device=DEVICE)
            terminal = torch.tensor([done], device=DEVICE)

            # store (s, a, r, s+1, bool) in D
            agent.store_transition((state, action, next_state, reward, terminal))

            # sample random minibatch of (st,at, r, st+1) from D
            minibatch = agent.sample_experience

            if minibatch is None:
                state = next_state
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
            state = next_state

            if terminal:
                break


if __name__ == "__main__":
    main()
