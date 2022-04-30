import os
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, config):
        self.x_label = "Episodes"
        self.y_label = "Last 100 average score"
        self.plot_file_name = self.get_file_name(config)

    def get_plot(self, average_latest_scores):
        plt.close("all")
        plt.plot(average_latest_scores)
        plt.title(self.plot_file_name)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.savefig(f"plot/{self.plot_file_name}.png")

    def get_file_name(self, config):
        method = config["method"]
        custom_name = ""
        if config["per"] == "True":
            custom_name += "_PER"
        if config["dueling"] == "True":
            custom_name += "_Dueling"
        if config["custom_name"]:
            custom_name += f"_{config['custom_name']}"

        file_name = f"plot_boxing-v5_{method}DQN{custom_name}"

        i = 1
        while os.path.exists(f"plot/{file_name}_{i}.png"):
            i += 1
    
        return f"{file_name}_{i}"
