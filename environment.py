from abc import abstractmethod
from typing import Any, Tuple

import numpy as np

class BaseEnvironment:
    @abstractmethod
    def start(self) -> Any:
        raise NotImplementedError('Expected `start` to be implemented')

    @abstractmethod
    def step(self, action: int) -> Tuple[float, Any, bool]:
        raise NotImplementedError('Expected `step` to be implemented')

class randomContextualBandit(BaseEnvironment):
    numContexts = None
    numArms = None

    means = None

    currentContext = None

    t = None

    def __init__(self):
        self.numContexts = 2
        self.numArms = 3
        self.means = np.random.rand(self.numContexts, self.numArms)
        self.stddev = np.random.rand(self.numContexts, self.numArms)/10
        #self.means = np.array([[0,.5,1],[.5,1,0]])
        #self.stddev = np.array([[.2,.2,.2],[.2,.2,.2]])

        print(self.means, "\n", self.stddev)
        self.t = 0

    def best_reward(self, context):
        return np.max(self.means[np.argmax(context)])

    def start(self):
        self.currentContext = np.random.randint(self.numContexts)

        context = np.zeros(self.numContexts)
        context[self.currentContext] = 1

        return context

    def step(self, action):
        self.t += 1

        if self.t % 20000 == 0 and self.t <= 40000:
            print("change", self.t)
            self.means = np.random.rand(self.numContexts, self.numArms)*2
            self.stddev = np.random.rand(self.numContexts, self.numArms)/10*2
            # new mean and stddev

        mean = self.means[self.currentContext, action]
        stddev = self.stddev[self.currentContext, action]
        reward = np.random.normal(mean, stddev)
        #reward = self.means[self.currentContext, action]

        self.currentContext = np.random.randint(self.numContexts)

        context = np.zeros(self.numContexts)
        context[self.currentContext] = 1

        return (reward, context, False)

    def agentParams(self):
        return {"numActions": self.numArms, "observationDims": self.numContexts}
