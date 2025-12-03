from __future__ import annotations

import click
import yaml

from cli.agent import agent
from cli.utils import (
    LEARNING_POLICIES,
)
import engine


@agent.command("create", help="Create a new agent with action, exploration, and learning policies.")
@click.option(
    "--name",
    "name",
    type=str,
    default=None,
    prompt="Agent name",
    required=False,
    help="Name of the agent to create.",
)
@click.option(
    "--action-policy",
    "action_policy",
    type=click.Choice(engine.agent_manager.list_action_policies()),
    default=engine.agent_manager.list_action_policies()[0],
    show_default=True,
    prompt="Action policy",
    help="Action selection policy to attach to the agent.",
)
@click.option(
    "--exploration-policy",
    "exploration_policy",
    type=click.Choice(engine.agent_manager.list_exploration_policies()),
    default=engine.agent_manager.list_exploration_policies()[0],
    show_default=True,
    prompt="Exploration policy",
    help="Exploration strategy to use.",
)
@click.option(
    "--learning-policy",
    "learning_policy",
    type=click.Choice(engine.agent_manager.list_learning_policies()),
    default=engine.agent_manager.list_learning_policies()[0],
    show_default=True,
    prompt="Learning policy",
    help="Learning update strategy.",
)
def create_agent(
    name: str | None,
    action_policy: str,
    exploration_policy: str,
    learning_policy: str,
) -> None:
    while not name or name in engine.list_agents():
        name = click.prompt("Agent name")

    if action_policy == "q_value" and learning_policy != "tabular_q":
        raise click.ClickException("Action policy 'q_value' requires learning policy 'tabular_q'.")

    payload = {
        "name": name,
        "action_policy": action_policy,
        "exploration_policy": exploration_policy,
        "learning_policy": learning_policy,
    }
    result = engine.create_agent(**payload)

    click.echo(
        f"Created agent '{name}' with action policy '{action_policy}'"
    )
