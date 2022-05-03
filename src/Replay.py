from collections import deque
import random

import numpy as np

class ReplayMemory(object):
    def __init__(self, transition_format, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)
        self.transition = transition_format

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class PriorityReplayMemory(object):
    #offset
    e = 0.01 
    #randomness factor 0 = random 1 = only highest prio
    a = 0.6
    #beta should anneal up to 1 over the duration of training
    beta = 0.4
    #value for annealing beta
    beta_increment_per_sampling = 0.0001

    def __init__(self, transition_format, capacity: int, offset, alpha, beta, beta_increment_per_sampling) -> None:
        #self.memory = PriorityQueue(maxsize=capacity) 
        self.transition = transition_format
        self.capacity = capacity

        self.priorities = deque(maxlen=capacity)

        #Initialize hyper parms
        self.e = offset
        self.a = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        self.tree = SumTree(capacity)
        

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
    
    def push(self, error, *args):
        p = self._get_priority(error)
        self.tree.add(p, *args)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        #updating importance Sampling Weights
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        #normailizing weight
        is_weight /= is_weight.max()

        return batch, is_weight, idxs


    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)    
        

#https://github.com/rlcode/per/blob/master/SumTree.py used for sumtree implementation
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])