import argparse


def build_parser(include_num_envs: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable cProfile around the episode loop")
    parser.add_argument("--model-path", default="q_model.pkl", help="Path to load/save the Q-table")
    parser.add_argument("--fresh", action="store_true", help="Start with a fresh model instead of loading")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run a rendered demo with the loaded agent")
    parser.add_argument("--episodes", type=int, default=5000000, help="Number of training episodes to run")
    parser.add_argument("--demos-per-run", type=int, default=1, help="Number of rendered demo episodes to run")
    if include_num_envs:
        parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    return parser
