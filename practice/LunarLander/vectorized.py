import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
import cProfile
import pstats
import sys

# Allow imports from the parent project directory.
PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from agents import TabularQAgent
from cli import build_parser



def run(
    profile_enabled: bool = False,
    model_path: Path = Path("q_model.pkl"),
    fresh: bool = False,
    eval_only: bool = False,
    total_episodes: int = 500000,
    num_envs: int = 1,
):
    rewards = []

    # Environment factory so we can reuse it for training and the demo run.
    def make_env(render_mode=None):
        def _thunk():
            return gym.make(
                "LunarLander-v3",
                continuous=False,
                gravity=-10.0,
                enable_wind=False,
                wind_power=15.0,
                turbulence_power=1.5,
                render_mode=render_mode,
            )
        return _thunk

    # Async to leverage multiple cores.
    env = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
    agent = TabularQAgent(env.single_observation_space, env.single_action_space)
    if not fresh and model_path.exists():
        agent.load(model_path)

    obs, info = env.reset()
    per_env_returns = np.zeros(num_envs, dtype=np.float64)
    episodes_finished = 0

    def rollout():
        nonlocal obs, episodes_finished, per_env_returns
        with tqdm.trange(total_episodes, desc="Episodes") as pbar:
            while episodes_finished < total_episodes:
                actions = np.empty(num_envs, dtype=int)
                for idx in range(num_envs):
                    actions[idx] = agent.select_action(env.single_action_space, obs[idx])

                next_obs, rewards_step, terminated, truncated, infos = env.step(actions)
                dones = np.logical_or(terminated, truncated)

                # Use final_observation for terminal transitions (vector env auto-resets).
                final_obs = infos.get("final_observation")

                for idx in range(num_envs):
                    if final_obs is not None and final_obs[idx] is not None:
                        obs_for_update = final_obs[idx]
                    else:
                        obs_for_update = next_obs[idx]
                    agent.update(
                        rewards_step[idx],
                        obs[idx],
                        int(actions[idx]),
                        obs_for_update,
                        bool(dones[idx]),
                    )

                per_env_returns += rewards_step

                if np.any(dones):
                    for idx in range(num_envs):
                        if dones[idx]:
                            episodes_finished += 1
                            rewards.append(per_env_returns[idx])
                            per_env_returns[idx] = 0.0
                            pbar.update(1)
                            if episodes_finished >= total_episodes:
                                break

                obs = next_obs

    if not eval_only:
        if profile_enabled:
            with cProfile.Profile() as pr:
                rollout()
            stats = pstats.Stats(pr)
            stats.sort_stats("cumtime").print_stats(40)
        else:
            rollout()

    env.close()

    # Render a single evaluation episode with the learned policy.
    demo_env = make_env(render_mode="human")()
    # Greedy run for the demo.
    agent.exploration_rate = 0.0
    demo_obs, _ = demo_env.reset()
    done = False
    demo_return = 0.0
    while not done:
        action = agent.select_action(demo_env.action_space, demo_obs)
        demo_obs, reward, terminated, truncated, _ = demo_env.step(action)
        demo_return += reward
        done = terminated or truncated
    print(f"Demo episode return: {demo_return}")
    demo_env.close()

    # Persist the learned table.
    if not eval_only:
        agent.save(model_path)

    if not eval_only:
        display_learning(rewards)


def display_learning(rewards):
    rewards_array = np.array(rewards)
    # Plot the rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_array)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot rewards with moving average
    plt.subplot(1, 2, 2)
    window_size = 100
    moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.title(f'Moving Average of Rewards (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = build_parser(include_num_envs=True).parse_args()
    run(
        profile_enabled=args.profile,
        model_path=Path(args.model_path),
        fresh=args.fresh,
        eval_only=args.eval_only,
        total_episodes=args.episodes,
        num_envs=args.num_envs,
    )
