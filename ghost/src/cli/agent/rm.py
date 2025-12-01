from __future__ import annotations

import click

from cli.agent import agent
import engine


@agent.command("rm", help="Delete an agent by name.")
@click.argument("name", required=False)
def delete_agent(name: str | None) -> None:
    existing = engine.list_agents()
    if not existing:
        raise click.ClickException("No agents to delete.")

    if not name:
        click.echo("Available agents:")
        for agent_name in existing:
            cfg = engine.describe_agent(agent_name)
            click.echo(
                f"- {agent_name}: action={cfg.get('action_policy')}, "
                f"exploration={cfg.get('exploration_policy', {}).get('type')}, "
                f"learning={cfg.get('learning_policy', {}).get('type')}"
            )
        name = click.prompt(
            "Agent to delete", type=click.Choice(existing)
        )

    engine.delete_agent(name)
    click.echo(f"Deleted agent '{name}'")
