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
    generator = None

    numContexts = None
    numArms = None

    means = None

    currentContext = None

    t = None

    def __init__(self, seed):
        self.t = 0

        self.generator = np.random.default_rng(seed)

        self.numContexts = 10
        self.numArms = 5

        self.randomize_env()

    def best_reward(self, context):
        return np.max(self.means[np.argmax(context)])

    def best_action(self, context):
        return np.argmax(self.means[np.argmax(context)])

    def gen_context(self):
        return self.generator.integers(self.numContexts)

    def randomize_env(self):
        self.means = self.generator.random((self.numContexts, self.numArms))*2
        self.stddev = self.generator.random((self.numContexts, self.numArms))/10*2

    def start(self):
        self.currentContext = self.gen_context()

        context = np.zeros(self.numContexts)
        context[self.currentContext] = 1

        return (context, self.best_action(context))

    def step(self, action):
        self.t += 1

        if self.t % 15000 == 0 and self.t <= 75000:
            print("change", self.t)
            # new mean and stddev
            self.randomize_env()
            print(self.means)

        mean = self.means[self.currentContext, action]
        stddev = self.stddev[self.currentContext, action]
        reward = self.generator.normal(mean, stddev)
        #reward = self.means[self.currentContext, action]

        self.currentContext = self.gen_context()

        context = np.zeros(self.numContexts)
        context[self.currentContext] = 1

        return (reward, context, False, self.best_action(context))

    def agentParams(self):
        return {"numActions": self.numArms, "observationDims": self.numContexts}
