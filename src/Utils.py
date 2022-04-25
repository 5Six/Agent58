import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_data(path: str) -> None:
    print(f"loading data from {path}...")


def nparray_to_tensor(nparray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    temp = torch.from_numpy(nparray).float().to(device)
    return temp[None, :]


class Plot:
    def __init__(self, config):
        self.x_label = "Episodes"
        self.y_label = "Last 100 average score"
        self.plot_save_path = self.get_save_path(config)

    def get_plot(self, average_latest_scores):
        plt.close("all")
        plt.plot(average_latest_scores)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.savefig(self.plot_save_path)

    def get_save_path(self, config):
        method = config['method']
        custom_name = config['custom_name']

        if custom_name:
            custom_name = "_" + custom_name
        save_path = f"plot/plot_boxing-v5_{method}DQN{custom_name}"

        i = 1
        while os.path.exists(f"{save_path}_{i}.png"):
            i += 1
    
        return f"{save_path}_{i}.png"
