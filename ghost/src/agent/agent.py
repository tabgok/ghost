from abc import abstractmethod
from agent.policies.action.policy import ActionPolicy
from agent.policies.action.policy import HumanActionPolicy
from agent.policies.learning.policy import LearningPolicy
from agent.policies.learning.policy import NoOpLearningPolicy
from agent.policies.exploration.policy import ExplorationPolicy
from agent.policies.exploration.policy import NoOpExplorationPolicy

class Agent:
    def __init__(self,
                 learning_policy: LearningPolicy,
                 action_policy: ActionPolicy,
                 exploration_policy: ExplorationPolicy) -> None:
        self.learning_policy = learning_policy()
        self.action_policy = action_policy()
        self.exploration_policy = exploration_policy()

    def select_action(self, action_space, observation_space, observation):
        action = self.action_policy.act(action_space, observation)
        action = self.exploration_policy.explore(action, action_space)
        return action

    def learn(self, prior_observation: any, observation: any, action: int, reward: float, done: bool) -> None:
        self.learning_policy.learn(prior_observation, observation, action, reward, done)
    
    def end_episode(self) -> None:
        pass

        


class _HumanAgent(Agent):
    def __init__(self):
        return super().__init__(
            learning_policy=NoOpLearningPolicy,
            action_policy=HumanActionPolicy,
            exploration_policy=NoOpExplorationPolicy)