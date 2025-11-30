from __future__ import annotations

import random
from typing import Any, Iterable, List

from agent.policies.action.policy import ActionPolicy
from agent.policies.learning.qlearning import TabularQLearningPolicy


class QValuePolicy(ActionPolicy):
    """Greedy action selection driven by a tabular Q-value source."""

    def __init__(self, learner: TabularQLearningPolicy) -> None:
        self.learner = learner

    def act(self, action_space: Any, observation: Any) -> Any:
        state_key = self.learner.state_key(observation)
        state_q = self.learner.q_table.get(state_key, {})
        actions = self._available_actions(action_space)
        if not actions:
            return self._fallback(action_space, actions)

        best_q = max((state_q.get(action, 0.0) for action in actions), default=0.0)
        best_actions = [action for action in actions if state_q.get(action, 0.0) == best_q]
        return random.choice(best_actions) if best_actions else self._fallback(action_space, actions)

    def _available_actions(self, action_space: Any) -> List[Any]:
        if hasattr(action_space, "available_actions"):
            try:
                return list(action_space.available_actions())
            except Exception:
                pass

        if hasattr(action_space, "n"):
            try:
                return list(range(int(action_space.n)))
            except Exception:
                pass

        if isinstance(action_space, (list, tuple, set)):
            return list(action_space)

        try:
            return list(action_space)
        except Exception:
            return []

    def _fallback(self, action_space: Any, actions: Iterable[Any] | None = None) -> Any:
        action_list = list(actions) if actions is not None else []
        if action_list:
            return random.choice(action_list)
        if hasattr(action_space, "sample"):
            return action_space.sample()
        raise RuntimeError("No available actions for QValuePolicy.")
