import os
import torch
import numpy as np
import matplotlib.pyplot as plt


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

class Plot:
    def __init__(self, method='', custom_name=''):
        self.x_label = "Episodes"
        self.y_label = "Last 100 average score"
        self.plot_save_path = self.get_save_path(method, custom_name)

    def get_plot(self, avg_score_history):
        plt.close("all")
        plt.plot(avg_score_history)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.savefig(self.plot_save_path)

    def get_save_path(self, method, custom_name):
        if custom_name:
            custom_name = "_" + custom_name
        save_path = f"plot/plot_boxing-v5_{method}DQN{custom_name}"

        i = 1
        while os.path.exists(f"{save_path}_{i}.png"):
            i += 1
        
        return f"{save_path}_{i}.png"
