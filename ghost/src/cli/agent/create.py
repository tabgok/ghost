from __future__ import annotations

import click
import yaml

from cli.agent import agent
from cli.utils import (
    ACTIONS,
    AGENTS_DIR,
    AGENT_EXT,
    LEARNING_POLICIES,
)

DEFAULT_ACTION_POLICY = "random"
DEFAULT_EXPLORATION_POLICY = "epsilon_greedy"
DEFAULT_LEARNING_POLICY = "noop"


@agent.command("create", help="Create a new agent with action, exploration, and learning policies.")
@click.argument("name", required=False)
@click.option(
    "--action-policy",
    "action_policy",
    type=click.Choice(ACTIONS),
    default=DEFAULT_ACTION_POLICY,
    show_default=True,
    prompt="Action policy",
    help="Action selection policy to attach to the agent.",
)
@click.option(
    "--exploration-policy",
    "exploration_policy",
    type=click.Choice([DEFAULT_EXPLORATION_POLICY]),
    default=DEFAULT_EXPLORATION_POLICY,
    show_default=True,
    prompt="Exploration policy",
    help="Exploration strategy to use.",
)
@click.option(
    "--epsilon-start",
    type=float,
    default=0.1,
    show_default=True,
    prompt="Epsilon start",
    help="Starting epsilon for epsilon-greedy exploration.",
)
@click.option(
    "--epsilon-min",
    type=float,
    default=0.01,
    show_default=True,
    prompt="Epsilon min",
    help="Minimum epsilon value.",
)
@click.option(
    "--epsilon-decay",
    type=float,
    default=0.99,
    show_default=True,
    prompt="Epsilon decay",
    help="Decay factor applied each step.",
)
@click.option(
    "--learning-policy",
    "learning_policy",
    type=click.Choice(LEARNING_POLICIES),
    default=DEFAULT_LEARNING_POLICY,
    show_default=True,
    prompt="Learning policy",
    help="Learning update strategy.",
)
def create_agent(
    name: str | None,
    action_policy: str,
    exploration_policy: str,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    learning_policy: str,
) -> None:
    if not name:
        name = click.prompt("Agent name")

    if action_policy == "q_value" and learning_policy != "tabular_q":
        raise click.ClickException("Action policy 'q_value' requires learning policy 'tabular_q'.")

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = AGENTS_DIR / f"{name}{AGENT_EXT}"
    while path.exists():
        click.echo(f"Agent '{name}' already exists at {path}")
        name = click.prompt("Enter a new agent name")
        path = AGENTS_DIR / f"{name}{AGENT_EXT}"

    payload = {
        "name": name,
        "action_policy": action_policy,
        "exploration_policy": {
            "type": exploration_policy,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        },
        "learning_policy": {"type": learning_policy},
    }
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False)

    click.echo(
        f"Created agent '{name}' with action policy '{action_policy}' at {path}"
    )
