from __future__ import annotations

import click

from cli.agent import agent as agent_group
from cli.run import run
from cli.train import train


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Ghost CLI for managing agents.",
)
def ghost() -> None:
    """Base Ghost CLI group."""


ghost.add_command(agent_group)
ghost.add_command(run)
ghost.add_command(train)


def main() -> None:
    ghost(prog_name="ghost")
