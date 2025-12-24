import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


from engine import environment_manager
from engine import agent_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)  # or WARNING




def run(agent_names: tuple[str], environment: str, episodes: int, plot: bool, show_progress: bool) -> None:
    logger.info(f"Running agents {agent_names} in environment {environment} for {episodes} episodes.")

    agents = []
    for agent in agent_names:
        logger.debug(f"Loading agent: {agent}")
        agent = agent_manager.AGENT_REGISTRY[agent]()
        agents.append(agent)
    # Load the environment
    logger.debug(f"Loading environment: {environment}")
    env = environment_manager.instantiate_environment(environment)
    episode_iterator = None
    if show_progress:
        episode_iterator = trange(1, episodes+1, desc="Training Episodes")
    else:
        episode_iterator = range(1, episodes+1)  # disable progress bar for now
    episode_returns: dict[str, list[float]] = {agent.name: [] for agent in agents}
    for _ in episode_iterator:
        # Reset environment to start a new episode
        observation, info = env.reset()
        observation = _discretize(observation)

        episode_over = False
        reward = 0
        steps = 0
        per_agent_return = [0.0 for _ in agents]
        turn = 0
        while not episode_over:
            agent = agents[turn]
            turn = (turn + 1) % len(agents)
            action = agent.select_action(env.action_space, observation)
            prior_observation = observation

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)
            observation = _discretize(observation)
            agent.learn(prior_observation, observation, action, reward, terminated or truncated)
            steps += 1

            per_agent_return[(turn - 1) % len(agents)] += reward
            episode_over = terminated or truncated
            logger.debug(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        for agent in agents:
            agent.end_episode()
        for idx, agent in enumerate(agents):
            episode_returns[agent.name].append(per_agent_return[idx])
    env.close()
    logger.debug(episode_returns)
    if plot:
        plot_learning_curve(episode_returns)


def plot_learning_curve(returns: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(1)
    plt.show(block=False)
    plt.pause(1)
    episodes = len(next(iter(returns.values())))
    window = max(10, episodes // 100)  # ~1% smoothing window
    for agent_name, trajectory in returns.items():
        traj = np.asarray(trajectory, dtype=float)
        if traj.size == 0:
            continue
        # scatter a sparse sample of raw returns to show variance
        stride = max(1, traj.size // 200)
        ax.scatter(np.arange(0, traj.size, stride), traj[::stride], s=8, alpha=0.2)
        if traj.size >= window:
            kernel = np.ones(window, dtype=float) / float(window)
            smooth = np.convolve(traj, kernel, mode="valid")
            x = np.arange(smooth.size) + window // 2
            ax.plot(x, smooth, label=f"{agent_name} (mean {window})")
        else:
            ax.plot(traj, label=agent_name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Training episode returns (smoothed)")
    ax.legend()
    manager = plt.get_current_fig_manager()
    if hasattr(manager, "window") and hasattr(manager.window, "wm_geometry"):
        manager.window.wm_geometry("1200x800")
        def _resize_to_window(event) -> None:
            if event.width <= 0 or event.height <= 0:
                return
            fig.set_size_inches(event.width / fig.dpi, event.height / fig.dpi, forward=True)
            fig.canvas.draw_idle()
        manager.window.bind("<Configure>", _resize_to_window)
    plt.show()


def _discretize(obs):
    """
    Convert a continuous 4D CartPole observation into a discrete state tuple.
    
    Parameters:
        obs (array-like): [cart position, cart velocity, pole angle, pole angular velocGity]
    
    Returns:
    """
    return obs
