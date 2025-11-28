import sys
from pathlib import Path
import unittest

import numpy as np

# Ensure src/ is on path for direct import
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.custom.tictactoe import _check_winner  # noqa: E402


class TestTicTacToeWinner(unittest.TestCase):
    def test_wins_and_draws(self):
        wins = {
            "row_p1": (
                np.array(
                    [
                        [1, 1, 1],
                        [0, 2, 0],
                        [0, 0, 2],
                    ]
                ),
                1,
                [(0, 0), (0, 1), (0, 2)],
            ),
            "row_p2": (
                np.array(
                    [
                        [2, 2, 2],
                        [1, 0, 1],
                        [0, 1, 0],
                    ]
                ),
                2,
                [(0, 0), (0, 1), (0, 2)],
            ),
            "col_p1": (
                np.array(
                    [
                        [1, 2, 0],
                        [1, 2, 0],
                        [1, 0, 2],
                    ]
                ),
                1,
                [(0, 0), (1, 0), (2, 0)],
            ),
            "col_p2": (
                np.array(
                    [
                        [1, 2, 0],
                        [0, 2, 1],
                        [1, 2, 0],
                    ]
                ),
                2,
                [(0, 1), (1, 1), (2, 1)],
            ),
            "diag_p1": (
                np.array(
                    [
                        [1, 2, 0],
                        [0, 1, 2],
                        [0, 0, 1],
                    ]
                ),
                1,
                [(0, 0), (1, 1), (2, 2)],
            ),
            "diag_p2": (
                np.array(
                    [
                        [2, 1, 0],
                        [0, 2, 1],
                        [0, 0, 2],
                    ]
                ),
                2,
                [(0, 0), (1, 1), (2, 2)],
            ),
            "anti_diag_p1": (
                np.array(
                    [
                        [0, 0, 1],
                        [0, 1, 2],
                        [1, 2, 2],
                    ]
                ),
                1,
                [(0, 2), (1, 1), (2, 0)],
            ),
            "anti_diag_p2": (
                np.array(
                    [
                        [1, 0, 2],
                        [0, 2, 0],
                        [2, 1, 1],
                    ]
                ),
                2,
                [(0, 2), (1, 1), (2, 0)],
            ),
        }
        for name, (board, marker, cells) in wins.items():
            with self.subTest(win=name):
                has_win, found_cells = _check_winner(board, marker)
                self.assertTrue(has_win)
                self.assertEqual(found_cells, cells)

        draws = [
            np.array([[1, 2, 1], [2, 1, 2], [2, 1, 2]]),
            np.array([[2, 1, 2], [2, 1, 1], [1, 2, 2]]),
        ]
        for idx, board in enumerate(draws):
            with self.subTest(draw=idx):
                has_win, cells = _check_winner(board, 1)
                self.assertFalse(has_win)
                self.assertIsNone(cells)
                has_win, cells = _check_winner(board, 2)
                self.assertFalse(has_win)
                self.assertIsNone(cells)


if __name__ == "__main__":
    unittest.main()
