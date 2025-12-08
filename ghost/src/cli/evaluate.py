from __future__ import annotations

import click

import engine

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
    prompt=False,
    type=click.Choice(engine.list_agents()),
    help="Agents for the environment, can be repeated for multiple agents.",
)
def evaluate(env_name: str, agents: tuple[str]) -> None:
    if not agents:
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

    engine.evaluate(agents, env_name)