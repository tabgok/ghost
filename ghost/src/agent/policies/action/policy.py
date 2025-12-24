from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from logging import getLogger
logger = getLogger(__name__)

ACTION_POLICY_REGISTRY: dict[str, ActionPolicy] = {}

def _register_action_policy(cls: type):
    ACTION_POLICY_REGISTRY[cls.__name__] = cls
    return cls

class ActionPolicy(ABC):
    @abstractmethod
    def act(self, action_space: Any, observation: Any, learned_values: Any) -> Any:
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
    def act(self, action_space: Any, observation: Any, learned_values: Any) -> Any:
        action = action_space.sample()
        return action


@_register_action_policy
class GreedyPolicy(ActionPolicy):
    def act(self, action_space: Any, observation: Any, learned_values: Any) -> Any:
        logger.info(f"GreedyPolicy selecting action with values: {learned_values}")
        if not learned_values:
            logger.warning("No learned values provided; using random action.")
            return action_space.sample()
        best_action = max(learned_values, key=learned_values.get)  # gives 6 for your example
        logger.info(f"GreedyPolicy selected action: {best_action}")
        return best_action