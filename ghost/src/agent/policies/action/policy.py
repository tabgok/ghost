from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ActionPolicy(ABC):
    @abstractmethod
    def act(self, action_space: Any, observation: Any) -> Any:
        """Select an action given the available action space and current observation."""


class HumanActionPolicy(ActionPolicy):
    def act(self, action_space: Any, observation: Any) -> Any:
        action = input(f"Observation: {observation}\nSelect action from {action_space.human_options}: ")
        return action