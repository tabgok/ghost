from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import tkinter as tk
import time

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
    render_mode: str | None = "human"

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __post_init__(self) -> None:
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self._board = np.zeros((3, 3), dtype=np.int8)
        self._done = False
        if self.render_mode not in self.metadata["render_modes"]:
            self.render_mode = None
        self._window: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._cell_size = 120
        self._margin = 10

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
        if self.render_mode == "human":
            self._render_tk(self._render_frame(), delay=True)

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
                if self.render_mode == "human":
                    self._render_tk(self._render_frame(), delay=True)

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
        frame = self._render_frame()
        if self.render_mode == "human":
            self._render_tk(frame, delay=False)
            return None
        if self.render_mode == "rgb_array":
            return frame
        return None

    def _render_frame(self):
        size = 240
        cell = size // 3
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)
        line_color = np.array([0, 0, 0], dtype=np.uint8)

        for i in range(1, 3):
            canvas[i * cell - 1 : i * cell + 1, :, :] = line_color
            canvas[:, i * cell - 1 : i * cell + 1, :] = line_color

        for r in range(3):
            for c in range(3):
                marker = self._board[r, c]
                if marker == 0:
                    continue
                y0, y1 = r * cell, (r + 1) * cell
                x0, x1 = c * cell, (c + 1) * cell
                sub = canvas[y0:y1, x0:x1]
                rr, cc = np.indices(sub.shape[:2])
                if marker == 1:
                    mask = (np.abs(rr - cc) <= 2) | (
                        np.abs((rr + cc) - (cell - 1)) <= 2
                    )
                    sub[mask] = np.array([220, 20, 60], dtype=np.uint8)
                elif marker == 2:
                    center = (cell - 1) / 2.0
                    dist = (rr - center) ** 2 + (cc - center) ** 2
                    radius = (cell * 0.35) ** 2
                    inner = (cell * 0.27) ** 2
                    ring = (dist <= radius) & (dist >= inner)
                    sub[ring] = np.array([65, 105, 225], dtype=np.uint8)
                canvas[y0:y1, x0:x1] = sub
        return canvas

    def _render_tk(self, frame: np.ndarray, delay: bool) -> None:
        try:
            if self._window is None:
                self._window = tk.Tk()
                self._window.title("TicTacToe")
                total_size = self._cell_size * 3 + self._margin * 2
                self._canvas = tk.Canvas(
                    self._window,
                    width=total_size,
                    height=total_size,
                    bg="white",
                    highlightthickness=0,
                )
                self._canvas.pack()
            canvas = self._canvas
            if canvas is None:
                return

            canvas.delete("all")
            offset = self._margin
            cs = self._cell_size

            # Grid lines
            for i in range(1, 3):
                x = offset + i * cs
                canvas.create_line(x, offset, x, offset + 3 * cs, width=2)
                canvas.create_line(offset, offset + i * cs, offset + 3 * cs, offset + i * cs, width=2)

            # Markers
            for r in range(3):
                for c in range(3):
                    marker = self._board[r, c]
                    x0 = offset + c * cs
                    y0 = offset + r * cs
                    x1 = x0 + cs
                    y1 = y0 + cs
                    if marker == 1:  # X
                        canvas.create_line(x0 + 10, y0 + 10, x1 - 10, y1 - 10, width=3, fill="#DC143C")
                        canvas.create_line(x0 + 10, y1 - 10, x1 - 10, y0 + 10, width=3, fill="#DC143C")
                    elif marker == 2:  # O
                        canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, width=3, outline="#4169E1")

            self._window.update_idletasks()
            self._window.update()
            if delay:
                time.sleep(0.5)
        except tk.TclError:
            # Headless environment; fallback to printing
            symbols = {0: ".", 1: "X", 2: "O"}
            lines = [" ".join(symbols[cell] for cell in row) for row in self._board]
            print("\n".join(lines))
