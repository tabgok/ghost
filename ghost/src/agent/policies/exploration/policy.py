from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any

EXPLORATION_POLICY_REGISTRY: dict[str, ExplorationPolicy] = {}

def _register_exploration_policy(cls: type):
    EXPLORATION_POLICY_REGISTRY[cls.__name__] = cls
    return cls

class ExplorationPolicy(ABC):
    """Determines whether to explore based on step counts or other signals."""

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""
        return {"type": self.__class__.__name__}


@_register_exploration_policy
class NoOpExplorationPolicy(ExplorationPolicy):
    """A policy that never explores."""

    def explore(self, action: Any, action_space: Any) -> Any:
        return action

@_register_exploration_policy
class EpsilonDecayExplorationPolicy(ExplorationPolicy):
    """An epsilon-greedy exploration policy with decay over time."""
    def __init__(self, initial_epsilon: float = 1.0, min_epsilon: float = 0.1, decay_rate: float = 0.99):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def explore(self, action: Any, action_space: Any) -> Any:
        if random.random() < self.epsilon:
            action = action_space.sample()
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        return action