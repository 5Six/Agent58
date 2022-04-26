import os
import matplotlib.pyplot as plt


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
        method = config["method"]
        custom_name = config["custom_name"]

        if custom_name:
            custom_name = "_" + custom_name
        save_path = f"plot/plot_boxing-v5_{method}DQN{custom_name}"

        i = 1
        while os.path.exists(f"{save_path}_{i}.png"):
            i += 1

        return f"{save_path}_{i}.png"
