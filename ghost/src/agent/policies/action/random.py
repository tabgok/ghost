from __future__ import annotations

import random
from typing import Any

from agent.policies.action.policy import ActionPolicy

class RandomPolicy(ActionPolicy):
    def act(self, action_space: Any, observation: Any) -> Any:
        return random.choice(action_space)
