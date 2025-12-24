from __future__ import annotations

import click

import engine

"""
@click.option(
    "--environment",
    "env_name",
    required=True,
    prompt=True,
    type=click.Choice(engine.environment_manager.list_environments()),
    default=engine.environment_manager.list_environments()[0],
    help="Name of the environment to be trained in.",
)
@click.option(
    "--episodes",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    prompt=True,
    help="Number of training episodes.",
)
"""

@click.command("run", help="Run an agent in a specified environment.")
def run() -> None:
    sample_run = {
        "environment": "TicTacToe",
        "agents": ("MonteCarloAgent", "RandomAgent"),
        "episodes": 100000,
        "render_mode": "None",
        "plot_learning_curve": False,
        "show_progress": True,
        "profile": False,
    }
    agents = ()

    env_name = sample_run["environment"]
    episodes = sample_run["episodes"]
    plot_results = sample_run["plot_learning_curve"]
    profile = sample_run["profile"]
    show_progress = sample_run["show_progress"]

    for agent in sample_run["agents"]:
        agents += (agent,)

    if profile:
        import cProfile
        import pstats
        from pathlib import Path

        stats_path = Path("profile.out")
        profiler = cProfile.Profile()
        profiler.runcall(engine.run, agents, env_name, episodes, plot=plot_results, show_progress=show_progress)
        profiler.dump_stats(stats_path)
        pstats.Stats(str(stats_path)).sort_stats("cumtime").print_stats(30)
    else:
        engine.run(agents, env_name, episodes, plot=plot_results, show_progress=show_progress)
