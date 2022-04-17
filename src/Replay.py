from collections import deque
import random


class ReplayMemory(object):
    def __init__(self, transition_format, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)
        self.transition = transition_format

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size) -> list[tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
