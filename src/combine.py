from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque


def main():
    mypath = "./Score_logs"
    fileNames = [f[:-4] for f in listdir(mypath) if isfile(join(mypath, f))] 

    all_scores = []

    for filename in fileNames:
        
        fullpath = mypath + "/" + filename +".txt"
        file = np.loadtxt(fullpath)
        
        latest_scores = deque(maxlen=100) # store the latest 100 scores
        average_latest_scores = []

        for row in file:
            latest_scores.append(row)
            average_latest_scores.append(np.mean(latest_scores))

        all_scores.append(average_latest_scores)
    
    colors = ["b","g","r","c","m","y"]
    x = np.arange(len(all_scores[0]))
    ax = plt.subplot(111)
    for i in range(len(all_scores)):
        ax.plot(x, all_scores[i], label=fileNames[i], color=colors[i], linewidth=0.8, markersize=2)

    ax.legend()

    plt.xlabel("Episode")
    plt.ylabel("Last 100 average scores")
    plt.title("Comparisons of learning curve across all DQN variations")
    plt.savefig(f"plot/test.png")

main()