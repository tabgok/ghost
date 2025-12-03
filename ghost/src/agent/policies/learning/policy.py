from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LearningPolicy(ABC):
    """Defines how an agent updates its internal state from experience."""

    @abstractmethod
    def learn(self, transition: Any) -> None:
        """Update internal state given a transition."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""


class NoLearningPolicy(LearningPolicy):
    """A policy that does not learn."""

    def learn(self, *args, **kwargs) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {"type": "NoLearningPolicy"}