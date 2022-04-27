from collections import deque
import random
import numpy as np


class ReplayMemory(object):
    def __init__(self, transition_format, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)
        self.transition = transition_format
        self.priorities = deque(maxlen=capacity)


    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(self.transition(*args))
        self.priorities.append(max(self.priorities, default=1))

  


    def get_importance(self, probabilities, beta =0.4):
        importance = 1/len(self.memory) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
        
    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.memory), batch_size)
        
        probs = np.array((self.priorities)) ** priority_scale
        probs /= probs.sum()
        
        sample_indices = random.choices(range(len(self.memory)), k=sample_size, weights=probs)
        samples = [self.memory[idx] for idx in sample_indices]
        
        total = len(self.memory)
        beta=0.4
        weights = (total * probs[sample_indices]) ** (-beta)
        weights /= weights.max()
        #weights = np.array(weights, dtype = np.float32)

        return samples, weights, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        count = 0
        for i in indices:
            i = int(i)
            self.priorities[i] = errors[count].item() + offset
            count+=1

    def __len__(self) -> int:
        return len(self.memory)

