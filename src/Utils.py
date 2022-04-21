import torch
import numpy as np


def load_data(path: str) -> None:
    print(f"loading data from {path}...")


def nparray_to_tensor(nparray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    temp = torch.from_numpy(nparray).float().to(device)
    return temp[None, :]


# def plot_scores():
#     plt.figure(2)
#     plt.clf()
#     scores_t = torch.tensor(scores, dtype=torch.float)
#     plt.title("Training...")
#     plt.xlabel("Episode")
#     plt.ylabel("Scores")
#     plt.plot(scores_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(scores_t) >= 100:
#         means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())
