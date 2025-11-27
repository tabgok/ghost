from __future__ import annotations

from pathlib import Path
import yaml
import click

BASE_DIR = Path.home() / ".ghost"
AGENTS_DIR = BASE_DIR / "agents"
AGENT_EXT = ".yaml"
DEFAULT_ACTION_POLICY = "random"


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Ghost CLI for managing agents.",
)
def ghost() -> None:
    """Base Ghost CLI group."""


@ghost.group(help="Agent-related commands.")
def agent() -> None:
    """Agent management subgroup."""


@agent.command("ls", help="List available agents.")
def list_agents() -> None:
    if not AGENTS_DIR.exists():
        click.echo("No agents found.")
        return

    agents = sorted(p.stem for p in AGENTS_DIR.glob(f"*{AGENT_EXT}") if p.is_file())
    if not agents:
        click.echo("No agents found.")
        return

    click.echo("Agents:")
    for name in agents:
        click.echo(f"- {name}")


@agent.command("create", help="Create a new agent (action policy only for now).")
@click.argument("name", required=False)
@click.option(
    "--action-policy",
    "action_policy",
    type=click.Choice([DEFAULT_ACTION_POLICY]),
    default=DEFAULT_ACTION_POLICY,
    show_default=True,
    prompt="Action policy",
    help="Action selection policy to attach to the agent.",
)
def create_agent(name: str | None, action_policy: str) -> None:
    if not name:
        name = click.prompt("Agent name")

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = AGENTS_DIR / f"{name}{AGENT_EXT}"
    if path.exists():
        raise click.ClickException(f"Agent '{name}' already exists at {path}")

    payload = {"name": name, "action_policy": action_policy}
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False)

    click.echo(
        f"Created agent '{name}' with action policy '{action_policy}' at {path}"
    )


def main() -> None:
    ghost(prog_name="ghost")
