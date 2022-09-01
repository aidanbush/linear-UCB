import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.lines import Line2D

def loadData(filename):
    jsonObj = {}

    with open(filename) as file:
        jsonObj = json.load(file)

    # return means, return stdev, regret means, regret stdev
    data = [[np.array(jsonObj["returns"]["mean"]), np.array(jsonObj["returns"]["stdev"]), np.array(jsonObj["regrets"]["mean"]), np.array(jsonObj["regrets"]["stdev"])]]

    return data

def plotData(data, labels):
    # plot steps
    plt.figure(figsize=(16,16))
    for i in range(len(data)):
        label = labels[i]
        plt.plot(data[i][0], label=label)
        plt.fill_between(np.arange(len(data[i][0])), data[i][0]-data[i][1], data[i][0]+data[i][1], alpha=0.5)

    plt.grid(linestyle='--')
    plt.title("steps")
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()
    #plt.savefig("step_plot.pdf")

    plt.figure(figsize=(16,16))

    # plot returns
    for i in range(len(data)):
        label = labels[i]
        plt.plot(data[i][2], label=label)
        plt.fill_between(np.arange(len(data[i][2])), data[i][2]-data[i][3], data[i][2]+data[i][3], alpha=0.5)

    plt.grid(linestyle='--')
    plt.title("return")
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()
    #plt.savefig("return_plot.pdf")

    plt.figure(figsize=(16,16))

    # plot regrets
    for i in range(len(data)):
        label = labels[i]
        plt.plot(data[i][4], label=label)
        plt.fill_between(np.arange(len(data[i][4])), data[i][4]-data[i][5], data[i][4]+data[i][5], alpha=0.5)

    plt.grid(linestyle='--')
    plt.title("regret")
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()
    #plt.savefig("regret_plot.pdf")

def plot():
    data = loadData("cppout")
    plotData(data, "run")

plot()
