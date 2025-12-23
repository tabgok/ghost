from __future__ import annotations

import click
import yaml
import gymnasium as gym

from agent.policies.action.random import RandomPolicy
from agent.policies.action.policy import HumanActionPolicy
from agent.policies.action.qvalue import QValuePolicy
from agent.policies.learning.qlearning import TabularQLearningPolicy
from envs.custom.tictactoe import TicTacToeEnv
import engine

DEFAULT_ACTION_POLICY = "random"
DEFAULT_EXPLORATION_POLICY = "epsilon_greedy"
DEFAULT_LEARNING_POLICY = "noop"
AVAILABLE_ENVS = ["tictactoe", "cartpole", "lunar_lander"]  # TODO: Refactor this, and the below, so there is a registry
LEARNING_POLICIES = [DEFAULT_LEARNING_POLICY, "tabular_q"]



def make_env(env_name: str, render_mode: str, two_players: bool = False):
    if env_name == "tictactoe":
        mode = None if render_mode == "none" else render_mode
        return TicTacToeEnv(render_mode=mode, opponent_random=not two_players)
    if env_name == "cartpole":
        return gym.make("CartPole-v1", render_mode=None if render_mode == "none" else render_mode)
    if env_name == "lunar_lander":
        return gym.make("LunarLander-v2", render_mode=None if render_mode == "none" else render_mode)
    raise click.ClickException(f"Unknown environment '{env_name}'")
