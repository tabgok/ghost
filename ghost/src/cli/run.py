from __future__ import annotations

import time
import click

from cli.utils import (
    AVAILABLE_ENVS,
)


@click.command("run", help="Run agents in an environment.")
@click.option(
    "--agent-a",
    "agent_a",
    required=False,
    help="Name of the first agent (will prompt if omitted).",
)
@click.option(
    "--agent-b",
    "agent_b",
    required=False,
    help="Name of the second agent (will prompt if omitted).",
)
@click.option(
    "--env",
    "env_name",
    type=click.Choice(AVAILABLE_ENVS),
    required=False,
    help="Environment to run (will prompt if omitted).",
)
@click.option(
    "--episodes",
    type=int,
    default=1,
    show_default=True,
    help="Number of episodes to run.",
)
@click.option(
    "--render",
    "render_mode",
    type=click.Choice(["human", "none", "rgb_array"]),
    default="human",
    show_default=True,
    help="Rendering mode during evaluation.",
)
def run(
    agent_a: str | None,
    agent_b: str | None,
    env_name: str | None,
    episodes: int,
    render_mode: str,
) -> None:
    pass