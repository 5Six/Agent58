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

  
    def get_probabilities(self, priority_scale):
        #print(self.priorities)
        scaled_priorities = np.array((self.priorities)) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities


    def get_importance(self, probabilities):
        importance = 1/len(self.memory) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
        
    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.memory), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.memory)), k=sample_size, weights=sample_probs)
        samples = []
        #random_samples = random.sample(self.memory, batch_size)
        for i in sample_indices:
            samples.append(self.memory[i])
        
        #samples = np.array(self.memory)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        count = 0
        for i in indices:
            i = int(i)
            self.priorities[i] = errors[count].item() + offset
            count+=1

    def __len__(self) -> int:
        return len(self.memory)
