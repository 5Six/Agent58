from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque

import plotly.graph_objects as go


def main():
    mypath = "./Results"
    fileNames = [f for f in listdir(mypath) if isfile(join(mypath, f))] 
    all_scores = []

    for filename in fileNames:
        print(filename)
        fullpath = mypath + "/" + filename
        file = np.loadtxt(fullpath)
        
        latest_scores = deque(maxlen=300) # store the latest 100 scores
        average_latest_scores = []

        for row in file:
            latest_scores.append(row)
            average_latest_scores.append(row)

        all_scores.append(average_latest_scores)
    
    # colors = ["b","g","r","c","m","y"]
    x = np.arange(len(all_scores[0]))

    # start plotting
    fig = go.Figure()
    for i in range(len(all_scores)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=all_scores[i],
                name=fileNames[i],
            )
        )

    fig.update_layout(
    title="Comparison of learning curves across all DQN variations",
    xaxis_title="Episode",
    yaxis_title="Last 300 average scores",
    legend_title="Experiment Name",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="Black"
        )
    )

    fig.write_html("plot/last300.html")


main()