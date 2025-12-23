from __future__ import annotations

import click
import gymnasium as gym

from envs.custom.tictactoe import TicTacToeEnv

AVAILABLE_ENVS = ["tictactoe", "cartpole", "lunar_lander"]


def make_env(env_name: str, render_mode: str, two_players: bool = False):
    if env_name == "tictactoe":
        mode = None if render_mode == "none" else render_mode
        return TicTacToeEnv(render_mode=mode, opponent_random=not two_players)
    if env_name == "cartpole":
        return gym.make("CartPole-v1", render_mode=None if render_mode == "none" else render_mode)
    if env_name == "lunar_lander":
        return gym.make("LunarLander-v2", render_mode=None if render_mode == "none" else render_mode)
    raise click.ClickException(f"Unknown environment '{env_name}'")
