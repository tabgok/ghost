from __future__ import annotations

from pathlib import Path
import yaml
import click

import gymnasium as gym

from agent.policies.action.random import RandomPolicy
from envs.custom.tictactoe import TicTacToeEnv

BASE_DIR = Path.home() / ".ghost"
AGENTS_DIR = BASE_DIR / "agents"
AGENT_EXT = ".yaml"
DEFAULT_ACTION_POLICY = "random"
DEFAULT_EXPLORATION_POLICY = "epsilon_greedy"
DEFAULT_LEARNING_POLICY = "noop"
AVAILABLE_ENVS = ["tictactoe", "cartpole", "lunar_lander"]


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
    agents = _list_agent_names()
    if not agents:
        click.echo("No agents found.")
        return

    click.echo("Agents:")
    for name in agents:
        click.echo(f"- {name}")


@agent.command("create", help="Create a new agent with action, exploration, and learning policies.")
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
    type=click.Choice([DEFAULT_LEARNING_POLICY]),
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


def main() -> None:
    ghost(prog_name="ghost")


@ghost.command("run", help="Run an agent in an environment.")
@click.option(
    "--agent",
    "agent_name",
    required=False,
    help="Name of the agent to run (will prompt if omitted).",
)
@click.option(
    "--env",
    "env_name",
    type=click.Choice(AVAILABLE_ENVS),
    required=False,
    help="Environment to run (will prompt if omitted).",
)
@click.option(
    "--episodes",
    type=int,
    default=1,
    show_default=True,
    help="Number of episodes to run.",
)
def run(agent_name: str | None, env_name: str | None, episodes: int) -> None:
    if not agent_name:
        existing = _list_agent_names()
        default_agent = existing[0] if existing else None
        agent_name = click.prompt(
            "Agent name",
            type=click.Choice(existing) if existing else str,
            default=default_agent,
            show_default=bool(default_agent),
        )
    if not env_name:
        env_name = click.prompt(
            "Environment", type=click.Choice(AVAILABLE_ENVS), default="tictactoe"
        )

    agent_path = AGENTS_DIR / f"{agent_name}{AGENT_EXT}"
    if not agent_path.exists():
        raise click.ClickException(f"Agent '{agent_name}' not found at {agent_path}")

    with agent_path.open(encoding="utf-8") as fp:
        agent_cfg = yaml.safe_load(fp) or {}

    env = _make_env(env_name)
    policy = _load_action_policy(agent_cfg.get("action_policy", DEFAULT_ACTION_POLICY))

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = policy.act(env.action_space, obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            steps += 1

        click.echo(
            f"Episode {ep}/{episodes}: steps={steps}, total_reward={total_reward:.3f}"
        )


def _make_env(env_name: str):
    if env_name == "tictactoe":
        return TicTacToeEnv()
    if env_name == "cartpole":
        return gym.make("CartPole-v1")
    if env_name == "lunar_lander":
        return gym.make("LunarLander-v2")
    raise click.ClickException(f"Unknown environment '{env_name}'")


def _load_action_policy(name: str):
    if name == "random":
        return RandomPolicy()
    raise click.ClickException(f"Unsupported action policy '{name}'")


def _list_agent_names() -> list[str]:
    if not AGENTS_DIR.exists():
        return []
    return sorted(p.stem for p in AGENTS_DIR.glob(f"*{AGENT_EXT}") if p.is_file())
