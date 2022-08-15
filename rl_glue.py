from agent import BaseAgent
from environment import BaseEnvironment


class RlGlue:
    def __init__(self, agent: BaseAgent, env: BaseEnvironment):
        self.environment = env
        self.agent = agent

        self.last_action: int = -1
        self.total_reward: float = 0.0
        self.num_steps: int = 0
        self.total_steps: int = 0
        self.num_episodes: int = 0
        self.regret: int = 0

    def start(self):
        self.num_steps = 0
        self.total_reward = 0

        s, best_action = self.environment.start()
        self.last_action = self.agent.start(s, best_action)

        return (s, self.last_action)

    def step(self):
        (reward, s, term, best_action) = self.environment.step(self.last_action)

        self.total_reward += reward
        self.regret = self.environment.best_reward(s) - reward

        self.num_steps += 1
        self.total_steps += 1
        if term:
            self.num_episodes += 1
            self.agent.end(reward)
            rsat = (reward, s, None, term)
        else:
            self.last_action = self.agent.step(reward, s, best_action)
            rsat = (reward, s, self.last_action, term)

        return rsat

    def runEpisode(self, max_steps: int = 0):
        is_terminal = False

        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        # even at episode cutoff, this still counts as completing an episode
        if not is_terminal:
            self.num_episodes += 1

        return is_terminal
