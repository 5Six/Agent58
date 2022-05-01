from torch import nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, dimension: int, output: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimension * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class Dueling_DQN(nn.Module):
    def __init__(self, dimension: int, output: int) -> None:
        super(Dueling_DQN, self).__init__()
        self.pre = nn.Linear(256, 256)

        self.fc1_adv = nn.Linear(256, 256)
        self.fc1_val = nn.Linear(256, 256)

        self.fc2_adv = nn.Linear(256, 256)
        self.fc2_val = nn.Linear(256, 256)

        self.fc3_adv = nn.Linear(256, 18)
        self.fc3_val = nn.Linear(256, 1)


    def forward(self, x):
        adv = F.relu(self.pre(x))
        val = F.relu(self.pre(x))

        adv = F.relu(self.fc1_adv(adv))
        val = F.relu(self.fc1_val(val))

        adv = F.relu(self.fc2_adv(adv))
        val = F.relu(self.fc2_val(val))

        adv = self.fc3_adv(adv)
        # val = self.fc3_val(val).expand(x.size(0), 18)
        val = self.fc3_val(val)

        # q = val + (adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 18))
        q = val + (adv - adv.mean())
        return q