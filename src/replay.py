from collections import deque

class ReplayBuffer:
    def __init__(self, max_size, keys):
        self.dict = {}
        for key in keys:
            self.dict[key] = deque(maxlen=max_size)  # creating empty list for each key
        self.max_size = max_size

    def append(self, sample):
        for i, key in enumerate(self.dict.keys()):
            self.dict[key].append(sample[i])

    def __len__(self):
        return len(self.dict['state'])

    def get(self, ids):
        sub_dict = {}
        for key in self.dict.keys():
            sub_dict[key] = [self.dict[key][i] for i in ids]
        return sub_dict