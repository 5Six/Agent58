import numpy as np
from Replay import ReplayMemory


class Agent:
    def __init__(
        self,
        alpha: float,
        epsilon: float,
        gamma: float,
        memory_capacity: int = 1000,
        batch_size: int = 32,
    ) -> None:

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size

        self.buffer = ReplayMemory(self.memory_capacity)
        self.action_value_network = 1  # initalise NN
        self.target_value_network = 2  # initalise NN

    def choose_action(self, action_space) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.choice(action_space)
        else:
            # replace with NN.
            action = np.random.randint(0, 18)  # change to action space

        return action

    def store_transition(self, transition: tuple) -> None:
        self.buffer.push(transition)

    def sample_experience(self):
        return self.buffer.sample(self.batch_size)

    def get_best_arg(self) -> int:
        return np.random.randint(0, 18)

    # def learn(self, experience) -> None:
    #     if len(self.buffer) < self.batch_size:
    #         return

    #     # expected return of current state using action_value_network

    #     # get target return from target_value_network

    #     # calc loss ????

    #     return
