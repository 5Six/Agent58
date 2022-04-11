import numpy as np
import Replay

class Agent:

    def __init__(e,g,a,c=1000) -> None:
        self.EPSILON = e
        self.GAMMA = g
        self.ALPHA = a
        self.CAPACITY = c

        self.buffer = Replay.ReplayMemory(self.CAPACITY)
        return
    
    def choose_action(self) -> int:
        # if np.random.random() > self.EPSILON:
        #     # add NN
        # else:
        #     #random

        return np.random.randint(0, 18)# change to action space
