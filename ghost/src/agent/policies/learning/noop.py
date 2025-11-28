from __future__ import annotations

from typing import Any

from agent.policies.learning.policy import LearningPolicy


class NoOpLearningPolicy(LearningPolicy):
    """A learning policy that performs no updates."""

    def learn(self, transition: Any) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"type": "noop"}
