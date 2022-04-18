import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_net(nn.Module):
    def __init__(self):
        super().__init__()
        h = 256
        self.fc1 = nn.Linear(256, h)
        self.fc2 = nn.Linear(h, h)
        # self.bn2 = nn.BatchNorm1d(h)
        self.fc3 = nn.Linear(h, 18)
        # self.bn3 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        q = F.relu(self.fc3(x))
        # q = self.bn3(q)
        return q

