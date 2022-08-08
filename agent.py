from abc import abstractmethod
from typing import Any

import numpy as np

class BaseAgent:
    @abstractmethod
    def start(self, observation: Any) -> int:
        raise NotImplementedError('Expected `start` to be implemented')

    @abstractmethod
    def step(self, reward: float, observation: Any) -> int:
        raise NotImplementedError('Expected `step` to be implemented')

    @abstractmethod
    def end(self, reward: float) -> None:
        raise NotImplementedError('Expected `end` to be implemented')


class LinUCB(BaseAgent):
    # c context dimensions
    # a number of actions
    # theta a x c matrix of weights
    # context vertival vector of size c
    # action ???

    reg = None # regularizer

    V = None
    observeDims = None
    numActions = None

    oldAction = None

    theta = None # vector c*a in length

    t = None

    def __init__(self, parameters):
        self.observeDims = parameters["observationDims"]
        self.numActions = parameters["numActions"]
        self.reg = parameters["regularizer"]

        # TODO move to start?
        self.theta = np.random.rand(self.observeDims * self.numActions)
        self.V = np.identity(self.observeDims * self.numActions) * self.reg
        self.b = np.zeros(self.observeDims * self.numActions)

        self.delta = parameters["delta"]
        self.t = 1

    def start(self, observation):
        return self.selectAction(observation)

    def step(self, reward, observation):
        self.t += 1

        self.updateTheta(reward)

        if (self.t + 1) % (250) == 0:
            print(self.theta)

        return self.selectAction(observation)

    def end(self, reward):
        self.updateTheta(reward)

    def selectAction(self, observation):
        # select new action
        # observation is a vertical vector of size c
        # get action set - create context from observations - just a vector
        context = np.concatenate([observation for _ in range(self.numActions)])
        # beta = root(lambda) + root(2 log(1 / delta) + d log(1 + (t - 1) / lambda * d))
        beta = np.sqrt(self.reg) + np.sqrt(2 * np.log(1/self.delta) + self.numActions * np.log(1 + (self.t-1)/(self.reg * self.numActions)))
        # what is d here observation dimensions?

        # compute UCB
        ucb = np.zeros(self.numActions)
        for action in range(self.numActions):
            # block out non action elements
            actionContext = context * [1 if i // self.observeDims == action else 0 for i in range(self.observeDims * self.numActions)]
            # ucb = transpose(context) dot theta + beta root(transpose(context) dot V^inverse dot context)
            ucb[action] = actionContext.dot(self.theta) + beta * np.sqrt(actionContext.dot(np.linalg.inv(self.V)).dot(actionContext))
            # wrong should add a vector since I am calculating all UCB at the same time -- action vector (context here) is wrong since it should be different per action
        # action = argmax(ucb)
        action = np.random.choice(np.flatnonzero(ucb==ucb.max()))
        actionContext = context * [1 if i // self.observeDims == action else 0 for i in range(self.observeDims * self.numActions)]
        # oldAction = one hot vector
        self.oldAction = actionContext
        #print("action o:", actionContext)
        # return action
        return action

    def updateTheta(self, reward):
        #print("action u:", self.oldAction, reward, "\n")
        # update old action
        # V = V + old_action outer product transpose(old_action)
        self.V = self.V + np.outer(self.oldAction, self.oldAction)
        # b = b + reward * old_action
        self.b = self.b + reward * self.oldAction
        # theta = V^inverse * b
        self.theta = np.linalg.inv(self.V).dot(self.b)
