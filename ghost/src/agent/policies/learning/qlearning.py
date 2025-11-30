from __future__ import annotations

from typing import Any, Hashable, Tuple
from collections import defaultdict

from agent.policies.learning.policy import LearningPolicy


class TabularQLearningPolicy(LearningPolicy):
    """Simple tabular Q-learning scaffold."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99) -> None:
        # Alpha is the learning rate which determines how much new information overrides old information.
        self.alpha = float(alpha)
        self.learning_rate = self.alpha
        # Gamma is the discount factor which determines the importance of future rewards compared to immediate rewards.
        self.gamma = float(gamma)
        self.discount_factor = self.gamma
        self.q_table: defaultdict[Hashable, defaultdict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def _state_key(self, observation: Any) -> Hashable:
        # Basic hashing: rely on Python's tuple conversion for discrete/array-like obs.
        try:
            return tuple(observation.flatten().tolist())  # type: ignore[attr-defined]
        except Exception:
            try:
                return tuple(observation)
            except Exception:
                return observation

    def learn(self, transition: Tuple[Any, int, float, Any, bool]) -> None:
        """Update Q-table given (state, action, reward, next_state, done)."""
        state, action, reward, next_state, done = transition
        s_key = self._state_key(state)
        ns_key = self._state_key(next_state)

        state_q = self.q_table[s_key]
        next_q = self.q_table[ns_key]

        best_next = max(next_q.values(), default=0.0)
        target = reward + (0.0 if done else self.gamma * best_next)

        current = state_q.get(action, 0.0)
        state_q[action] = current + self.alpha * (target - current)

    def state_key(self, observation: Any) -> Hashable:
        """Public wrapper so action policies can reuse the hashing logic."""
        return self._state_key(observation)

    def snapshot(self) -> dict:
        return {
            "type": "tabular_q",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_table": {state: dict(actions) for state, actions in self.q_table.items()},
        }

    @classmethod
    def from_snapshot(cls, payload: dict) -> "TabularQLearningPolicy":
        policy = cls(
            alpha=payload.get("alpha", 0.1),
            gamma=payload.get("gamma", 0.99),
        )
        raw_table = payload.get("q_table", {}) or {}
        table: defaultdict[Hashable, defaultdict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for state, actions in raw_table.items():
            inner = defaultdict(float)
            inner.update(actions or {})
            table[state] = inner
        policy.q_table = table
        return policy
