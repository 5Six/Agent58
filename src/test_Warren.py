import Net
import torch
import numpy as np
# net = Net.Net(128,18)
# net2 = Net.Dueling_DQN(128,18)
# net = net.float()
# net.to("cuda")
# net2.to("cuda")
# t = torch.tensor(np.zeros([32,256]), device="cuda")
# with torch.no_grad():
#     net.eval()
#     net2.eval()
#     a = net(t.float())
#     b = net2(t.float())
# print(a.shape)
# print(b.shape)

a = torch.tensor([[1, 2, 3, 4],
                  [5, 1, 7, 8]])
print(a.shape)
print(a.max(0))
