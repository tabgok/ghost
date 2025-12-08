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
    def learn(self, transition: Any) -> None:
        """Update internal state given a transition."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any episode-specific state."""

    @abstractmethod
    def episode_end(self, reward: Any) -> None:
        """Handle end of episode updates."""

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""
        return {"type": self.__class__.__name__}


@_register_learning_policy
class NoOpLearningPolicy(LearningPolicy):
    """A policy that does not learn."""

    def learn(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass

    def episode_end(self, reward: Any) -> None:
        pass

@_register_learning_policy
class MonteCarloLearningPolicy(LearningPolicy):
    """A simple Monte Carlo learning policy."""
    def __init__(self):
        self.q_table = defaultdict(lambda: 0.8)  # Set high initial Q-values
        self.states = []

    def reset(self):
        self.states = []

    def learn(self, transition: Any) -> None:
        # Store the transitions and rewards
        self.states.append(transition)

    def episode_end(self, reward) -> None:
        """Handle end of episode updates."""
        pass