from abc import abstractmethod
from agent.policies.action.random import RandomPolicy

class Agent:
    def __init__(self, update_policy, action_policy: ActionPolicy, observation_space, action_space) -> None:
        self.update = update_policy
        self.act = action_policy
        self.observation_space = observation_space
        self.action_space = action_space


