from __future__ import annotations

import click


@click.group(help="Agent-related commands.")
def agent() -> None:
    """Agent management subgroup."""


# Import subcommands so they register with the group.
from cli.agent import ls, create, rm  # noqa: E402,F401
