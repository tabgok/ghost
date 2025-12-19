from abc import abstractmethod
from agent.policies.action.policy import ActionPolicy
from agent.policies.action.policy import ACTION_POLICY_REGISTRY
from agent.policies.learning.policy import LearningPolicy
from agent.policies.learning.policy import LEARNING_POLICY_REGISTRY
from agent.policies.exploration.policy import ExplorationPolicy
from agent.policies.exploration.policy import EXPLORATION_POLICY_REGISTRY


class Agent:
    def __init__(self,
                name: str,
                learning_policy: LearningPolicy,
                action_policy: ActionPolicy,
                exploration_policy: ExplorationPolicy) -> None:
        self.name = name
        self.learning_policy = learning_policy()
        self.action_policy = action_policy()
        self.exploration_policy = exploration_policy()

    def select_action(self, action_space, observation):
        learned_values = self.learning_policy.values(observation, action_space)
        action = self.action_policy.act(action_space, observation, learned_values)
        action = self.exploration_policy.explore(action, action_space)
        return action

    def learn(self, prior_observation: any, observation: any, action: int, reward: float, done: bool) -> None:
        self.learning_policy.learn(prior_observation, observation, action, reward, done)
    
    def end_episode(self) -> None:
        self.learning_policy.end_episode()

    def reset(self) -> None:
        self.learning_policy.reset()


    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "learning_policy": self.learning_policy.snapshot(),
            "action_policy": self.action_policy.snapshot(),
            "exploration_policy": self.exploration_policy.snapshot(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict):
        name = snapshot["name"]

        learning_policy_type = snapshot["learning_policy"]["type"]
        learning_policy_cls = LEARNING_POLICY_REGISTRY[learning_policy_type]

        action_policy_type = snapshot["action_policy"]["type"]
        action_policy_cls = ACTION_POLICY_REGISTRY[action_policy_type]

        exploration_policy_type = snapshot["exploration_policy"]["type"]
        exploration_policy_cls = EXPLORATION_POLICY_REGISTRY[exploration_policy_type]

        return cls(
            name=name,
            learning_policy=learning_policy_cls,
            action_policy=action_policy_cls,
            exploration_policy=exploration_policy_cls,
        )