import Net
import torch

print(1)
net = Net.Net()

t = torch.tensor([1]*128)

net.forward(t.float())
