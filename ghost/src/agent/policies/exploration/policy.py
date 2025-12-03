from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

EXPLORATION_POLICY_REGISTRY: dict[str, ExplorationPolicy] = {}

def _register_exploration_policy(cls: type):
    EXPLORATION_POLICY_REGISTRY[cls.__name__] = cls
    return cls

class ExplorationPolicy(ABC):
    """Determines whether to explore based on step counts or other signals."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""


@_register_exploration_policy
class NoOpExplorationPolicy(ExplorationPolicy):
    """A policy that never explores."""

    def explore(self, action: Any, action_space: Any) -> Any:
        return action

    def snapshot(self) -> dict[str, Any]:
        return {"type": "NoExplorationPolicy"}