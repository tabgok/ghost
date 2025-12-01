from abc import abstractmethod
from agent.policies.action.policy import ActionPolicy
from agent.policies.learning.policy import LearningPolicy
from agent.policies.exploration.policy import ExplorationPolicy

class Agent:
    def __init__(self,
                 learning_policy: LearningPolicy,
                 action_policy: ActionPolicy,
                 exploration_policy: ExplorationPolicy,
                 observation_space,
                 action_space) -> None:
        self.learning = learning_policy
        self.act = action_policy
        self.observation_space = observation_space
        self.action_space = action_space
        self.exploration_policy = exploration_policy


class _HumanAgent(Agent):
    def __init__(self):
        return super().__init__(
            learning_policy=None,
            action_policy=None,
            exploration_policy=None,
            observation_space=None,
            action_space=None)