from __future__ import annotations

import click
import yaml
import gymnasium as gym

from agent.policies.action.random import RandomPolicy
from agent.policies.action.human import HumanPolicy
from agent.policies.action.qvalue import QValuePolicy
from agent.policies.learning.qlearning import TabularQLearningPolicy
from envs.custom.tictactoe import TicTacToeEnv
import engine

DEFAULT_ACTION_POLICY = "random"
DEFAULT_EXPLORATION_POLICY = "epsilon_greedy"
DEFAULT_LEARNING_POLICY = "noop"
AVAILABLE_ENVS = ["tictactoe", "cartpole", "lunar_lander"]  # TODO: Refactor this, and the below, so there is a registry
LEARNING_POLICIES = [DEFAULT_LEARNING_POLICY, "tabular_q"]



def prompt_for_agent(label: str) -> str:
    existing = engine.list_agents()
    choices = existing + ["human"]
    default_agent = existing[0] if existing else "human"
    return click.prompt(
        f"{label} name",
        type=click.Choice(choices),
        default=default_agent,
        show_default=bool(default_agent),
    )


def load_action_policy(agent_cfg: dict):
    action_type = agent_cfg.get("action_policy", DEFAULT_ACTION_POLICY)
    if action_type == "random":
        return RandomPolicy()
    if action_type == "human":
        return HumanPolicy()
    if action_type == "q_value":
        learning_cfg = agent_cfg.get("learning_policy", {})
        if learning_cfg.get("type") != "tabular_q":
            raise click.ClickException("Q-value policy requires a tabular_q learning snapshot on the agent.")
        learner = TabularQLearningPolicy.from_snapshot(learning_cfg)
        return QValuePolicy(learner)
    raise click.ClickException(f"Unsupported action policy '{action_type}'")




def make_env(env_name: str, render_mode: str, two_players: bool = False):
    if env_name == "tictactoe":
        mode = None if render_mode == "none" else render_mode
        return TicTacToeEnv(render_mode=mode, opponent_random=not two_players)
    if env_name == "cartpole":
        return gym.make("CartPole-v1", render_mode=None if render_mode == "none" else render_mode)
    if env_name == "lunar_lander":
        return gym.make("LunarLander-v2", render_mode=None if render_mode == "none" else render_mode)
    raise click.ClickException(f"Unknown environment '{env_name}'")
