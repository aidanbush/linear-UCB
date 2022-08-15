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

def runTest(numRuns, numEpisodes, agent, env, parameters, testParams, report=False, seed=None):
    if seed == None:
        seed = int(time.time() % 10000 * 1000)

    # seed np
    np.random.seed(int(time.time() % 10000 * 1000))

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
        steps, returns, regrets = runAgent(numEpisodes, agent, env, parameters, testParams, report=report, seed=seed+i*numEpisodes)
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

def runAgent(numEpisodes, agentClass, envClass, parameters, testParams={"maxSteps":MAX_STEPS}, report=False, seed=None):
    if seed == None:
        seed = int(time.time() % 10000 * 1000)

    # create env
    env = envClass(seed)
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


    return steps, returns, regrets

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
    seed = int(time.time() % 10000 * 1000)

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
            data.append(runTest(numRuns, numEpisodes, agent, env, agentSweepParams, testParams, report=True, seed=seed))

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
            data.append(runTest(numRuns, numEpisodes, agent, env, agentSweepParams, testParams, report=True, seed=seed))

    # plot
    plotData(data, labels)

def basicTest():
    numRuns = 30
    numEpisodes = 12000
    #agents = [a.Random]
    #parameters = [{}]
    #agents = [a.BestAction, a.LinUCB]
    #parameters = [{}, {"regularizer": 1, "delta": 1.0}]
    #agents = [a.BestAction, a.LinUCB, a.LinUCB, a.LinUCB, a.LinUCB]
    #parameters = [{}, {"regularizer": 2, "delta": 1.0}, {"regularizer": 1, "delta": 1.0}, {"regularizer": .5, "delta": 1.0}, {"regularizer": .1, "delta": 1.0}]
    # no difference between regularizers
    agents = [a.BestAction, a.LinUCB, a.LinUCB, a.LinUCB, a.LinUCB]
    parameters = [{}, {"regularizer": 1, "delta": 1.0}, {"regularizer": 1, "delta": .5}, {"regularizer": 1, "delta": .1}, {"regularizer": 1, "delta": .05}]
    #agents = [a.BestAction, a.LinUCB, a.LinUCB, a.LinUCB]
    #agents = [a.BestAction, a.LinUCB]
    #parameters = [{}, {"regularizer": 1, "delta": 1}]
    env = e.randomContextualBandit
    #parameters = {"gamma": 1, "alpha": 0.1, "epsilon": 0.1, "n_steps": 5}
    testParams = {"algType": EPISODIC, "maxSteps":10}

    data = []
    labels = []

    seed = int(time.time() % 10000 * 1000)

    for i in range(len(agents)):
        labels.append(agents[i].__name__ + " " + str(i))
        agent = agents[i]
        params = parameters
        if isinstance(parameters, list):
            params = parameters[i]
        data.append(runTest(numRuns, numEpisodes, agent, env, params, testParams, report=True, seed=seed))

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
