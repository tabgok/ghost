from __future__ import annotations

import time
import click

import engine

@click.command("train", help="Train an agent in a specified environment.")
@click.option(
    "--environment",
    "env_name",
    required=True,
    prompt=True,
    type=click.Choice(engine.list_environments()),
    default=engine.list_environments()[0],
    help="Name of the environment to be trained in.",
)
@click.option(
    "--episodes",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    prompt=True,
    help="Number of training episodes.",
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    required=False,
    prompt=True,
    type=click.Choice(engine.list_agents()),
    default=(engine.list_agents()[0],),
    help="Agents for the environment, can be repeated for multiple agents.",
)
def train(env_name: str, agents: tuple[str], episodes: int) -> None:
    engine.train(agents, env_name, episodes)