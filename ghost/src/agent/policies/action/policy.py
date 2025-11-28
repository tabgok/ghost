from __future__ import annotations
from abc import abstractmethod


from abc import ABC, abstractmethod
from typing import Any


class ActionPolicy(ABC):
    @abstractmethod
    def act(self, action_space: Any, observation: Any) -> Any:
        """Select an action given the available action space and current observation."""
