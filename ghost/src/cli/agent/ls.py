from __future__ import annotations

import click

from cli.agent import agent
from cli.utils import list_agent_names, read_agent_config


@agent.command("ls", help="List available agents.")
def list_agents() -> None:
    agents = list_agent_names()
    if not agents:
        click.echo("No agents found.")
        return

    click.echo("Agents:")
    for name in agents:
        cfg = read_agent_config(name)
        click.echo(
            f"- {name}: action={cfg.get('action_policy')}, "
            f"exploration={cfg.get('exploration_policy', {}).get('type')}, "
            f"learning={cfg.get('learning_policy', {}).get('type')}"
        )
