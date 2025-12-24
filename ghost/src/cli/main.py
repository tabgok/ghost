from __future__ import annotations

import click

from cli.run import run 

from logging import basicConfig, DEBUG
FORMATTER = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
basicConfig(level=DEBUG, format=FORMATTER)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Ghost CLI for managing agents.",
)
def ghost() -> None:
    """Base Ghost CLI group."""


ghost.add_command(run)


def main() -> None:
    ghost(prog_name="ghost")
