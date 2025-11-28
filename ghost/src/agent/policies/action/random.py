from __future__ import annotations

import random
from typing import Any

import numpy as np

from agent.policies.action.policy import ActionPolicy


class RandomPolicy(ActionPolicy):
    def act(self, action_space: Any, observation: Any) -> Any:
        if hasattr(action_space, "sample"):
            return action_space.sample()
        return random.choice(action_space)
