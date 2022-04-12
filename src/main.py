from typing import Final
import gym
import ale_py
import Replay
import Net

from Agent import Agent


def main() -> None:

    ALPHA: Final = 0.9
    GAMMA: Final = 0.1
    EPSILON: Final = 0.1
    TOTAL_EPISODE_COUNT: Final = 100
    BATCH_SIZE: Final = 20
    MEMORY_CAPACITY: Final = 1000

    env = gym.make("ALE/Boxing-ram-v5")
    agent = Agent(ALPHA, EPSILON, GAMMA, MEMORY_CAPACITY, BATCH_SIZE)

    for _ in range(TOTAL_EPISODE_COUNT):
        done = False
        state = env.reset()

        while not done:
            action = agent.choose_action(env.action_space)
            next_state, reward, done, _ = env.step(action)

            if done:
                next_state = "Terminal"

            # store (st,at, r, st+1) in D
            agent.store_transition((state, action, next_state, reward))

            # sample random minibatch of (st,at, r, st+1) from D
            minibatch = agent.sample_experience()

            for i in minibatch:
                yj = 0
                if i.next_state == "Terminal":
                    yj = i.reward
                else:
                    yj = i.reward + agent.gamma * agent.get_best_arg()

                # perform gradient descent

            # if time_step % C == 0: theta2 = theta1
            state = next_state


if __name__ == "__main__":
    main()
