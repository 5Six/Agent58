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


class Dueling_DQN(nn.Module):
    def __init__(self):
        super(Dueling_DQN, self).__init__()
        h = 256
        self.fc1_adv = nn.Linear(256, h)
        self.fc1_val = nn.Linear(256, h)

        self.fc2_adv = nn.Linear(h, h)
        self.fc2_val = nn.Linear(h, h)

        self.fc3_adv = nn.Linear(256, 18)
        self.fc3_val = nn.Linear(256, 1)

        # self.relu = nn.ReLU()

    def forward(self, x):
        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = F.relu(self.fc2_adv(x))
        val = F.relu(self.fc2_val(x))

        adv = self.fc3_adv(adv)
        val = self.fc3_val(val).expand(x.size(0), 18)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 18)
        return x