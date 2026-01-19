import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cProfile
import pstats
from pathlib import Path
import sys

# Allow imports from the parent project directory.
PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from agents import TabularQAgent
from cli import build_parser


def main(
    profile_enabled: bool = False,
    model_path: Path = Path("q_model.pkl"),
    fresh: bool = False,
    eval_only: bool = False,
    demos_per_run: int = 1,
    total_episodes: int = 1000000,
):
    episodes = total_episodes
    display_count = 10

    render_mode = None
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
            enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode=render_mode)
    rewards = []
    #agent = RandomAgent(env.observation_space, env.action_space)
    agent = TabularQAgent(env.observation_space, env.action_space)
    if not fresh and model_path.exists():
        agent.load(model_path)
    last_exploration_rate = 0
    bucket_reward = 0

    global demos_so_far
    demos_so_far = 0
    def demo_episode():
        global demos_so_far
        print(f"Starting demo episode {demos_so_far + 1} of {demos_per_run}")
        if demos_so_far <= demos_per_run:
            return
        demos_so_far += 1
        demo_env = gym.make(
            "LunarLander-v3",
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
            render_mode="human",
        )
        agent.exploration_rate = 0.0
        demo_obs, _ = demo_env.reset()
        done = False
        demo_return = 0.0
        while not done:
            action = agent.select_action(demo_env.action_space, demo_obs)
            demo_obs, reward, terminated, truncated, _ = demo_env.step(action)
            demo_return += reward
            done = terminated or truncated
        demo_env.close()

    if eval_only:
        if demos_per_run > demos_so_far:
            demo_episode()
        return

    def rollout():
        nonlocal env, last_exploration_rate, bucket_reward
        for episode in tqdm.trange(episodes, desc="Episodes"):
            episode_reward = 0
            env = gym.make(
                "LunarLander-v3",
                continuous=False,
                gravity=-10.0,
                enable_wind=False,
                wind_power=15.0,
                turbulence_power=1.5,
                render_mode=render_mode,
            )

            last_observation, info = env.reset()

            episode_over = False

            while not episode_over:
                last_action = agent.select_action(env.action_space, last_observation)
                current_observation, reward, terminated, truncated, info = env.step(last_action)
                episode_over = terminated or truncated
                episode_reward += reward
                agent.update(reward, last_observation, last_action, current_observation, episode_over)
                last_observation = current_observation
            if display_count and (episode % (episodes // 10) == 0):
                agent.exploration_rate = last_exploration_rate
                #print(f"Evaluation run reward: {episode_reward}, final reward: {reward}, agent memsize: {len(agent.state_action.keys())}, exploration rate: {agent.exploration_rate}")
            rewards.append(episode_reward)
            bucket_reward += episode_reward
            env.close()
        demo_episode()

    if profile_enabled:
        with cProfile.Profile() as pr:
            rollout()
        stats = pstats.Stats(pr)
        stats.sort_stats("cumtime").print_stats(40)
    else:
        rollout()
    
    display_learning(rewards)
    agent.save(model_path)


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
    args = build_parser().parse_args()
    main(
        profile_enabled=args.profile,
        model_path=Path(args.model_path),
        fresh=args.fresh,
        eval_only=args.eval_only,
        demos_per_run=args.demos_per_run,
        total_episodes=args.episodes,
    )
