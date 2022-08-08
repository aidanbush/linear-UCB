import environment as e
import agent as a
from rl_glue import RlGlue

import numpy as np
import torch
import statistics
import time
import code
import random

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.lines import Line2D

MAX_STEPS = 10000
# types
EPISODIC="episodic"
CONTINUOUS="continuous"

def runTest(numRuns, numEpisodes, agent, env, parameters, testParams, report=False):
    stepsData = []
    stepsMeans = np.zeros(numEpisodes)
    stepsStdev = np.zeros(numEpisodes)

    returnsData = []
    returnsMeans = np.zeros(numEpisodes)
    returnsStdev = np.zeros(numEpisodes)

    regretsData = []
    regretsMeans = np.zeros(numEpisodes)
    regretsStdev = np.zeros(numEpisodes)

    for i in range(numRuns):
        steps, returns, regrets = runAgent(numEpisodes, agent, env, parameters, testParams, report=report)
        if report:
            print("Run", i)
        stepsData.append(steps)
        returnsData.append(returns)
        regretsData.append(regrets)

    # for each episode
    for i in range(numEpisodes):
        steps = []
        returns = []
        regrets = []
        # for each run
        for j in range(numRuns):
            steps.append(stepsData[j][i])
            returns.append(returnsData[j][i])
            regrets.append(regretsData[j][i])
        stepsMeans[i] = statistics.mean(steps)
        returnsMeans[i] = statistics.mean(returns)
        regretsMeans[i] = statistics.mean(regrets)
        if numRuns > 1:
            stepsStdev[i] = statistics.stdev(steps)
            returnsStdev[i] = statistics.stdev(returns)
            regretsStdev[i] = statistics.stdev(regrets)
        else:
            stepsStdev[i] = 0
            returnsStdev[i] = 0
            regretsStdev[i] = 0

    return stepsMeans, stepsStdev, returnsMeans, returnsStdev, regretsMeans, regretsStdev

def runAgent(numEpisodes, agentClass, envClass, parameters, testParams={"maxSteps":MAX_STEPS}, report=False):
    # seed np
    np.random.seed(int(time.time() % 10000 * 1000))

    # create env
    env = envClass()
    parameters.update(env.agentParams())

    # create agent
    agent = agentClass(parameters)

    # create glue object
    glue = RlGlue(agent, env)

    steps = []
    returns = []
    regrets = []

    # run test
    for i in range(numEpisodes):
        glue.runEpisode(max_steps=testParams["maxSteps"])
        if report:
            print("Episode", i, glue.num_steps, "steps")
        steps.append(glue.num_steps)
        returns.append(glue.total_reward)
        regrets.append(glue.regret)
    #agent.printValues() # for testing value and action value values
    #agent.printWeights()
    #plotActions(agent, env, 40)


    return steps, returns, regrets

def plotActions(agent, environment, resolution):
    stateRanges = environment.agentParams()["stateFormat"]
    # if not continuous min is 0
    if type(stateRanges[0]) == int:
        stateRanges = [(0, s) for s in stateRanges]

    numStates = len(stateRanges)
    numActions = environment.agentParams()["numActions"]

    actionColours = plt.cm.get_cmap("hsv", numActions+1)
    nonActionColour = (0.0, 0.0, 0.0, 1.0)

    print(stateRanges)

    # assume stateRange >= 2
    if numStates < 2:
        # plot line
        return

    # create figure with len(stateRange) * len(stateRange) plots
    fig, ax = plt.subplots(numStates, numStates)

    # for each pair of environment states
    stateSampleIterations = 200
    for s1 in range(numStates):
        print("s1",s1)
        for s2 in range(numStates):
            print("s2",s2)
            s1Min = stateRanges[s1][0]
            s1Max = stateRanges[s1][1]
            s2Min = stateRanges[s2][0]
            s2Max = stateRanges[s2][1]

            imshowList = np.zeros((resolution+1, resolution+1, 4))
            # loop through values
            for i in range(resolution+1):
                if s1 == s2:
                    break
                for j in range(resolution+1):
                    state = [None for _ in range(numStates)]

                    s1Val = s1Min + (i/resolution)*(s1Max-s1Min)
                    s2Val = s2Min + (j/resolution)*(s2Max-s2Min)

                    state[s1] = s1Val
                    state[s2] = s2Val

                    action = agent.greedyAction(state, stateSampleIterations)
                    colour = nonActionColour

                    if action != agent.nullAction:# unupdated
                        colour = actionColours(action)

                    # copy colour
                    for imI in range(4):
                        imshowList[i][j][imI] = colour[imI]
                    #print("[{:.2f},{:.2f}]".format(state[0],state[1]), end=",")
            #print()
            ax[s1][s2].imshow(imshowList)
            ax[s1][s2].xaxis.tick_top()
            ax[s1][s2].xaxis.set_label_position('top')

            ax[s1][s2].set_xlabel(s1)
            ax[s1][s2].set_ylabel(s2)

    lines = [Line2D([0], [0], color=colour, lw=4) for colour in [nonActionColour]+[actionColours(i) for i in range(numActions)]]
    ax[0][0].legend(lines, ["none"] + list(range(numActions)))
    plt.show()

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

def parameterSweep(numRuns, numEpisodes, agent, env, agentParams, testParams, numTests, grid=True):
    if grid:
        # GRID sweep
        # create dictionary of ranges
        sweepRanges = {}
        for key in agentParams.keys():
            if isinstance(agentParams[key], list):
                sweepRanges[key] = agentParams[key]

        print(sweepRanges)
        data = []
        labels = []

        # for each num numTests^testParams
        keys = list(sweepRanges.keys())
        for i in range(numTests**len(keys)):
            # create parameters for test i
            agentSweepParams = agentParams.copy()
            for j in range(len(keys)):
                key = keys[j]
                val = i // numTests**j % numTests / (numTests - 1)
                val = val * (sweepRanges[key][1] - sweepRanges[key][0]) + sweepRanges[key][0]
                agentSweepParams[key] = val

            # run test
            labels.append(agent.__name__ + " " + str(agentSweepParams))

            print("Sweep:", len(labels), "/", numTests**len(keys), labels[-1])
            data.append(runTest(numRuns, numEpisodes, agent, env, agentSweepParams, testParams, report=True))

    else:
        # random sweep
        # create a dictionary of sweep ranges
        sweepRanges = {}
        for key in agentParams.keys():
            if isinstance(agentParams[key], list):
                sweepRanges[key] = agentParams[key]

        data = []
        labels = []

        # for numTests run random parameter sweeps
        for _ in range(numTests):
            # create parameters for the test
            agentSweepParams = agentParams.copy()
            for key in sweepRanges:
                agentSweepParams[key] = np.random.random() * (sweepRanges[key][1] - sweepRanges[key][0]) + sweepRanges[key][0]

            # run test
            labels.append(agent.__name__ + " " + str(agentSweepParams))

            print("Sweep:", len(labels), "/", numTests, labels[-1])
            data.append(runTest(numRuns, numEpisodes, agent, env, agentSweepParams, testParams, report=True))

    # plot
    plotData(data, labels)

def basicTest():
    numRuns = 10
    numEpisodes = 12000
    agents = [a.LinUCB]#[a.LinUCB, a.LinUCB, a.LinUCB]
    parameters = [{"regularizer": 1, "delta": 1.0}, {"regularizer": 2, "delta": 1.0}, {"regularizer": .5, "delta": 1.0}]#[{"regularizer": 1, "delta": 1.0}, {"regularizer": 1, "delta": .5}, {"regularizer": 1, "delta": .05}, ]
    env = e.randomContextualBandit
    #parameters = {"gamma": 1, "alpha": 0.1, "epsilon": 0.1, "n_steps": 5}
    testParams = {"algType": EPISODIC, "maxSteps":10}

    data = []
    labels = []

    for i in range(len(agents)):
        labels.append(agents[i].__name__ + " " + str(i))
        agent = agents[i]
        params = parameters
        if isinstance(parameters, list):
            params = parameters[i]
        data.append(runTest(numRuns, numEpisodes, agent, env, params, testParams, report=True))

    plotData(data, labels)

def sweepTest():
    numRuns = 5
    numEpisodes = 400
    agent = None #a.
    env = None #e.
    # test parameters [low,high]
    parameters = {}
    testParams = {"algType": EPISODIC, "maxSteps":1000}

    parameterSweep(numRuns, numEpisodes, agent, env, parameters, testParams, 3, grid=False)

def main():
    torch.set_num_threads(torch.get_num_threads()*2-2)
    basicTest()
    #sweepTest()
    return

main()
