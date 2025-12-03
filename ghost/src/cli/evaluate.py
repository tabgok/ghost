from __future__ import annotations

import time
import click

import engine
from engine import environment_manager

@click.command("evaluate", help="Evaluate an agent in a specified environment.")
@click.option(
    "--environment",
    "env_name",
    required=True,
    prompt=True,
    type=click.Choice(engine.environment_manager.list_environments()),
    default=engine.environment_manager.list_environments()[0],
    help="Name of the environment to be evaluated in.",
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
def evaluate(env_name: str, agents: tuple[str]) -> None:
    if len(agents) != environment_manager.describe_environment(env_name).get("num_agents", 1):
        agents = ()

    for _ in range(environment_manager.describe_environment(env_name).get("num_agents", 1)):
        promt_for_agent = click.prompt(
            f"Select agent for environment '{env_name}'",
            type=click.Choice(engine.list_agents()),
            default=engine.list_agents()[0],
        )
        agents += (promt_for_agent,)

    engine.evaluate(agents, env_name)