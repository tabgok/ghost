from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExplorationPolicy(ABC):
    """Determines whether to explore based on step counts or other signals."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""


class NoExplorationPolicy(ExplorationPolicy):
    """A policy that never explores."""

    def explore(self, action: Any, action_space: Any) -> Any:
        return action

    def snapshot(self) -> dict[str, Any]:
        return {"type": "NoExplorationPolicy"}