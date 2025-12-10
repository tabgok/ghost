from __future__ import annotations

import time
import click

import engine
from engine import environment_manager

@click.command("train", help="Train an agent in a specified environment.")
@click.option(
    "--environment",
    "env_name",
    required=True,
    prompt=True,
    type=click.Choice(engine.environment_manager.list_environments()),
    default=engine.environment_manager.list_environments()[0],
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
def train(env_name: str, episodes: int) -> None:
    agents = ()

    env_cfg = engine.environment_manager.describe_environment(env_name)
    num_agents = env_cfg.get("metadata", {}).get("agent_count", 1)

    undefined_agents = num_agents - len(agents)
    for i in range(undefined_agents):
        promt_for_agent = click.prompt(
            f"Select agent {i+1} of {undefined_agents} for environment '{env_name}'",
            type=click.Choice(engine.list_agents()),
            default=engine.list_agents()[0],
        )
        agents += (promt_for_agent,)

    engine.train(agents, env_name, episodes)