from __future__ import annotations

from typing import Any, Iterable, List
import click

from agent.policies.action.policy import ActionPolicy


class HumanPolicy(ActionPolicy):
    """Prompt-driven action policy with minimal assumptions about the action space."""

    def act(self, action_space: Any, observation: Any) -> Any:
        options = self._enumerate_actions(action_space)
        if options:
            return self._prompt_choice(options, action_space, observation)

        # Fallback: ask for raw input and rely on contains/sample.
        value = click.prompt("Enter action", type=str)
        try:
            parsed = int(value)
        except ValueError:
            parsed = value
        if hasattr(action_space, "contains") and not action_space.contains(parsed):
            click.echo("Action not valid for this space; choosing random sample.")
            return action_space.sample() if hasattr(action_space, "sample") else parsed
        return parsed

    def _enumerate_actions(self, action_space: Any) -> List[Any]:
        # Custom available actions
        if hasattr(action_space, "available_actions"):
            try:
                return list(action_space.available_actions())
            except Exception:
                pass
        # Gymnasium Discrete
        if hasattr(action_space, "n"):
            return list(range(int(action_space.n)))
        # Simple iterable spaces
        if isinstance(action_space, (list, tuple, set)):
            return list(action_space)
        try:
            # Attempt to materialize a small iterable
            return list(action_space)
        except Exception:
            return []

    def _prompt_choice(self, options: Iterable[Any], action_space: Any, observation: Any) -> Any:
        options_list = list(options)
        display = ", ".join(str(opt) for opt in options_list)
        maybe_board = self._format_board(observation, options_list)
        if maybe_board:
            click.echo(maybe_board)
        while True:
            value = click.prompt(f"Choose action ({display})", type=str)
            try:
                parsed = int(value)
            except ValueError:
                parsed = value
            if parsed in options_list:
                return parsed
            if hasattr(action_space, "contains") and action_space.contains(parsed):
                return parsed
            click.echo(f"Invalid action '{value}'. Try again.")

    def _format_board(self, observation: Any, options: List[Any]) -> str | None:
        try:
            import numpy as np

            board = np.asarray(observation)
            if board.shape != (3, 3):
                return None
            symbols = {0: ".", 1: "X", 2: "O"}
            lines = []
            for r in range(3):
                row = []
                for c in range(3):
                    idx = r * 3 + c
                    if board[r, c] == 0 and idx in options:
                        row.append(str(idx))
                    else:
                        row.append(symbols.get(int(board[r, c]), "?"))
                lines.append(" ".join(row))
            return "Board (empty cells numbered):\n" + "\n".join(lines)
        except Exception:
            return None
