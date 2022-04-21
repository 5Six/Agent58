from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dimension: int, output: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimension, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
