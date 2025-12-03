from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

ACTION_POLICY_REGISTRY: dict[str, ActionPolicy] = {}

def _register_action_policy(cls: type):
    ACTION_POLICY_REGISTRY[cls.__name__] = cls
    return cls

class ActionPolicy(ABC):
    @abstractmethod
    def act(self, action_space: Any, observation: Any) -> Any:
        """Select an action given the available action space and current observation."""
    
    def snapshot(self) -> dict[str, Any]:
        """Return a serializable representation of this policy."""
        return {"type": self.__class__.__name__}


@_register_action_policy
class HumanActionPolicy(ActionPolicy):
    def act(self, action_space: Any, observation: Any) -> Any:
        action = input(f"Observation: {observation}\nSelect action from {action_space.human_options}: ")
        return action

    
@_register_action_policy
class RandomActionPolicy(ActionPolicy):
    def act(self, action_space: Any, observation: Any) -> Any:
        action = action_space.sample()
        return action