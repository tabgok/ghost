from __future__ import annotations

import click

from cli.agent import agent
import engine


@agent.command("ls", help="List available agents.")
def list_agents() -> None:
    agents = engine.list_agents()
    if not agents:
        click.echo("No agents found.")
        return

    click.echo("Agents:")
    for name in agents:
        cfg = engine.describe_agent(name)
        click.echo(
            f"- {name}: action={cfg.get('action_policy')}, "
            f"exploration={cfg.get('exploration_policy', {}).get('type')}, "
            f"learning={cfg.get('learning_policy', {}).get('type')}"
        )
