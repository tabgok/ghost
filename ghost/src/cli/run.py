from __future__ import annotations

import time
import click

from cli.utils import (
    AVAILABLE_ENVS,
    load_action_policy_from_file,
    make_env,
    prompt_for_agent,
)


@click.command("run", help="Run agents in an environment.")
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
        agent_a = prompt_for_agent("Agent A")
    if not agent_b:
        agent_b = prompt_for_agent("Agent B")
    if not env_name:
        env_name = click.prompt(
            "Environment", type=click.Choice(AVAILABLE_ENVS), default="tictactoe"
        )

    env = make_env(env_name, render_mode, two_players=True)
    policy_a = load_action_policy_from_file(agent_a)
    policy_b = load_action_policy_from_file(agent_b)

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
