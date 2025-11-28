from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _check_winner(board: np.ndarray, marker: int) -> bool:
    diagonals = np.stack([np.diag(board), np.diag(np.fliplr(board))])
    lines = np.concatenate([board, board.T, diagonals], axis=0)
    return np.any(np.all(lines == marker, axis=1))


@dataclass
class TicTacToeEnv(gym.Env):
    """Minimal Gymnasium Tic-Tac-Toe environment with a random opponent.

    - Agent is marker 1, opponent is marker 2, empty is 0.
    - Observations: 3x3 int grid.
    - Actions: Discrete(9) representing cell index (row-major).
    - Rewards: +1 win, -1 loss, 0 draw/ongoing, -0.5 invalid move (episode ends).
    """

    opponent_random: bool = True

    metadata = {"render_modes": ["human"]}

    def __post_init__(self) -> None:
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self._board = np.zeros((3, 3), dtype=np.int8)
        self._done = False

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._board.fill(0)
        self._done = False
        return self._board.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            return self._board.copy(), 0.0, True, True, {}

        row, col = divmod(int(action), 3)
        reward = 0.0
        terminated = False
        truncated = False

        # Invalid move ends episode with penalty
        if self._board[row, col] != 0:
            reward = -0.5
            terminated = True
            self._done = True
            return self._board.copy(), reward, terminated, truncated, {}

        # Agent move
        self._board[row, col] = 1
        if _check_winner(self._board, 1):
            reward = 1.0
            terminated = True
            self._done = True
            return self._board.copy(), reward, terminated, truncated, {}

        if not (self._board == 0).any():
            terminated = True
            self._done = True
            return self._board.copy(), reward, terminated, truncated, {}

        # Opponent move: pick random available cell
        if self.opponent_random:
            empties = np.argwhere(self._board == 0)
            if len(empties) > 0:
                idx = self.np_random.integers(len(empties))
                r, c = empties[idx]
                self._board[r, c] = 2

        if _check_winner(self._board, 2):
            reward = -1.0
            terminated = True
            self._done = True
            return self._board.copy(), reward, terminated, truncated, {}

        if not (self._board == 0).any():
            terminated = True
            self._done = True

        return self._board.copy(), reward, terminated, truncated, {}

    def render(self):
        board_str = "\n".join(
            " ".join({0: ".", 1: "X", 2: "O"}[cell] for cell in row)
            for row in self._board
        )
        print(board_str)
