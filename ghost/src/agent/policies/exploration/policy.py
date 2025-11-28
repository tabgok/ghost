from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExplorationPolicy(ABC):
    """Determines whether to explore based on step counts or other signals."""

    @abstractmethod
    def should_explore(self, step: int) -> bool:
        """Return True if the agent should take an exploratory action."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""
