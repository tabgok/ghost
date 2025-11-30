from __future__ import annotations

import click

from cli.agent import agent
from cli.utils import list_agent_names, read_agent_config, AGENTS_DIR, AGENT_EXT


@agent.command("rm", help="Delete an agent by name.")
@click.argument("name", required=False)
def delete_agent(name: str | None) -> None:
    existing = list_agent_names()
    if not existing:
        raise click.ClickException("No agents to delete.")

    if not name:
        click.echo("Available agents:")
        for agent_name in existing:
            cfg = read_agent_config(agent_name)
            click.echo(
                f"- {agent_name}: action={cfg.get('action_policy')}, "
                f"exploration={cfg.get('exploration_policy', {}).get('type')}, "
                f"learning={cfg.get('learning_policy', {}).get('type')}"
            )
        name = click.prompt(
            "Agent to delete", type=click.Choice(existing)
        )

    path = AGENTS_DIR / f"{name}{AGENT_EXT}"
    if not path.exists():
        raise click.ClickException(f"Agent '{name}' not found at {path}")
    path.unlink()
    click.echo(f"Deleted agent '{name}'")
