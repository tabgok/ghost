from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.policies.exploration.policy import ExplorationPolicy


@dataclass
class EpsilonGreedyExploration(ExplorationPolicy):
    """Epsilon-greedy exploration with decay to a minimum epsilon."""

    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float

    def __post_init__(self) -> None:
        self._epsilon = float(self.epsilon_start)
        self.epsilon_min = float(self.epsilon_min)
        self.epsilon_decay = float(self.epsilon_decay)

    def should_explore(self, step: int) -> bool:
        # Basic exponential decay each call; caller provides monotonically increasing step.
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        from random import random

        return random() < self._epsilon

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "epsilon_greedy",
            "epsilon": self._epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
