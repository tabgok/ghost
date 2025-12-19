from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

LEARNING_POLICY_REGISTRY: dict[str, LearningPolicy] = {}

def _register_learning_policy(cls: type):
    LEARNING_POLICY_REGISTRY[cls.__name__] = cls
    return cls

class LearningPolicy(ABC):
    """Defines how an agent updates its internal state from experience."""

    @abstractmethod
    def learn(self, prior_observation, observation, action, reward, done) -> None:
        """Update internal state given a transition."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any episode-specific state."""

    @abstractmethod
    def end_episode(self) -> None:
        """Handle end of episode updates."""

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""
        return {"type": self.__class__.__name__}


@_register_learning_policy
class NoOpLearningPolicy(LearningPolicy):
    """A policy that does not learn."""

    def values(self, observation, action_space) -> dict[int, float]:
        """Return empty values for all actions."""
        return {}
    
    def learn(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass

    def end_episode(self) -> None:
        pass


@_register_learning_policy
class MonteCarloLearningPolicy(LearningPolicy):
    """A simple Monte Carlo learning policy."""
    def __init__(self):
        self.q_table = defaultdict(float)
        self.state_history = []
        self.alpha = 0.1 # Learning rate, which is a proxy for "average" so we don't have to keep track of N
        self.gamma = 0.8 # Discount factor, which is a weight on future rewards

    def reset(self):
        self.state_history = []

    def learn(self, prior_observation, observation, action, reward, done) -> None:
        # Store the transitions and rewards
        prior_obs_hash = hash(prior_observation.tobytes())
        self.state_history.append((prior_obs_hash, action, reward))
    
    def values(self, observation, action_space) -> dict[int, float]:
        """Return the learned values for each action in the given observation."""
        obs_hash = hash(observation.tobytes())
        action_values = {}
        for action in action_space.available_actions():
            entry = (obs_hash, action)
            action_values[action] = self.q_table[entry]
        return action_values

    def end_episode(self) -> None:
        """Handle end of episode updates."""
        G = 0
        for prior_observation, last_action, reward in reversed(self.state_history):
            G = reward + self.gamma * G
            entry = (prior_observation, last_action)
            cur_value = self.q_table[entry]
            self.q_table[entry] = cur_value + self.alpha * (G - cur_value)
        self.reset()