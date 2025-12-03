from __future__ import annotations

from abc import ABC, abstractmethod
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

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""
        return {"type": self.__class__.__name__}


@_register_learning_policy
class NoOpLearningPolicy(LearningPolicy):
    """A policy that does not learn."""

    def learn(self, *args, **kwargs) -> None:
        pass