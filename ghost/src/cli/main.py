from __future__ import annotations

from pathlib import Path
import yaml
import click
import time

import gymnasium as gym

from agent.policies.action.random import RandomPolicy
from agent.policies.action.human import HumanPolicy
from envs.custom.tictactoe import TicTacToeEnv

BASE_DIR = Path.home() / ".ghost"
AGENTS_DIR = BASE_DIR / "agents"
AGENT_EXT = ".yaml"
DEFAULT_ACTION_POLICY = "random"
DEFAULT_EXPLORATION_POLICY = "epsilon_greedy"
DEFAULT_LEARNING_POLICY = "noop"
AVAILABLE_ENVS = ["tictactoe", "cartpole", "lunar_lander"]
ACTIONS = [DEFAULT_ACTION_POLICY, "human"]


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


@ghost.command("run", help="Run agents in an environment.")
@click.option(
    "--agent-a",
    "agent_a",
    required=False,
    help="Name of the first agent (will prompt if omitted).",
)
@click.option(
    "--agent-b",
    "agent_b",
    required=False,
    help="Name of the second agent (will prompt if omitted).",
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
@click.option(
    "--render",
    "render_mode",
    type=click.Choice(["human", "none", "rgb_array"]),
    default="human",
    show_default=True,
    help="Rendering mode during evaluation.",
)
def run(
    agent_a: str | None,
    agent_b: str | None,
    env_name: str | None,
    episodes: int,
    render_mode: str,
) -> None:
    if not agent_a:
        agent_a = _prompt_for_agent("Agent A")
    if not agent_b:
        agent_b = _prompt_for_agent("Agent B")
    if not env_name:
        env_name = click.prompt(
            "Environment", type=click.Choice(AVAILABLE_ENVS), default="tictactoe"
        )

    env = _make_env(env_name, render_mode, two_players=True)
    policy_a = _load_action_policy_from_file(agent_a)
    policy_b = _load_action_policy_from_file(agent_b)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        current = 0  # 0 for agent A (marker 1), 1 for agent B (marker 2)
        rewards = {1: 0.0, 2: 0.0}

        while not (terminated or truncated):
            policy = policy_a if current == 0 else policy_b
            marker = 1 if current == 0 else 2
            action = policy.act(env.action_space, obs)
            obs, reward, terminated, truncated, _info = env.step(action, marker=marker)
            rewards[marker] += reward
            steps += 1
            current = 1 - current
            if render_mode == "human" and hasattr(env, "render"):
                env.render()

        winner = env.winner
        outcome = (
            "draw"
            if winner is None
            else f"{'Agent A' if winner == 1 else 'Agent B'} (marker {winner})"
        )
        click.echo(
            f"Episode {ep}/{episodes}: steps={steps}, winner={outcome}, "
            f"rewards: Agent A={rewards[1]:.2f}, Agent B={rewards[2]:.2f}"
        )
        if render_mode == "human":
            time.sleep(2.0)


def _make_env(env_name: str, render_mode: str, two_players: bool = False):
    if env_name == "tictactoe":
        mode = None if render_mode == "none" else render_mode
        return TicTacToeEnv(render_mode=mode, opponent_random=not two_players)
    if env_name == "cartpole":
        return gym.make("CartPole-v1", render_mode=None if render_mode == "none" else render_mode)
    if env_name == "lunar_lander":
        return gym.make("LunarLander-v2", render_mode=None if render_mode == "none" else render_mode)
    raise click.ClickException(f"Unknown environment '{env_name}'")


def _load_action_policy(name: str):
    if name == "random":
        return RandomPolicy()
    if name == "human":
        return HumanPolicy()
    raise click.ClickException(f"Unsupported action policy '{name}'")


def _list_agent_names() -> list[str]:
    if not AGENTS_DIR.exists():
        return []
    return sorted(p.stem for p in AGENTS_DIR.glob(f"*{AGENT_EXT}") if p.is_file())


def _prompt_for_agent(label: str) -> str:
    existing = _list_agent_names()
    choices = existing + ["human"]
    default_agent = existing[0] if existing else "human"
    return click.prompt(
        f"{label} name",
        type=click.Choice(choices),
        default=default_agent,
        show_default=bool(default_agent),
    )


def _load_action_policy_from_file(agent_name: str):
    if agent_name.lower() == "human":
        return HumanPolicy()
    agent_path = AGENTS_DIR / f"{agent_name}{AGENT_EXT}"
    if not agent_path.exists():
        raise click.ClickException(f"Agent '{agent_name}' not found at {agent_path}")
    with agent_path.open(encoding="utf-8") as fp:
        agent_cfg = yaml.safe_load(fp) or {}
    return _load_action_policy(agent_cfg.get("action_policy", DEFAULT_ACTION_POLICY))
