import time
from logging import getLogger
import tqdm


from agent.agent import Agent
from engine import environment_manager
from engine import agent_manager

logger = getLogger(__name__)


### --- TRAINING --- ###
def train(agent_names: tuple[str], environment: str, episodes: int) -> None:
    logger.info(f"Training agents {agent_names} in environment {environment} for {episodes} episodes.")

    agents = []
    for agent in agent_names:
        logger.debug(f"Loading agent: {agent}")
        agent = agent_manager.AGENT_REGISTRY[agent]()
        agents.append(agent)
    # Load the environment
    logger.debug(f"Loading environment: {environment}")
    env = environment_manager.instantiate_environment(environment)
    returns = _loop(agents, env, episodes=episodes, progress_bar=True)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        logger.warning("Could not import matplotlib/numpy to plot returns: %s", exc)
    else:
        plt.figure()
        window = max(10, episodes // 100)  # ~1% smoothing window
        for agent_name, trajectory in returns.items():
            traj = np.asarray(trajectory, dtype=float)
            if traj.size == 0:
                continue
            # scatter a sparse sample of raw returns to show variance
            stride = max(1, traj.size // 200)
            plt.scatter(np.arange(0, traj.size, stride), traj[::stride], s=8, alpha=0.2)
            if traj.size >= window:
                kernel = np.ones(window, dtype=float) / float(window)
                smooth = np.convolve(traj, kernel, mode="valid")
                x = np.arange(smooth.size) + window // 2
                plt.plot(x, smooth, label=f"{agent_name} (mean {window})")
            else:
                plt.plot(traj, label=agent_name)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Training episode returns (smoothed)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    env = environment_manager.instantiate_environment(environment, render_mode="human")
    getLogger(("root")).setLevel("DEBUG")
    _loop(agents, env, episodes=1, progress_bar=False)


def evaluate(agent_names: tuple[str], environment: str) -> None:
    logger.info(f"Evaluating agents {agent_names} in environment {environment}.")

    # Load the agent(s)
    agents = []
    for agent in agent_names:
        logger.debug(f"Loading agent: {agent}")
        agent = agent_manager.AGENT_REGISTRY[agent]()
        agents.append(agent)

    # Load the environment
    logger.debug(f"Loading environment: {environment}")
    env = environment_manager.instantiate_environment(environment, render_mode="human")
    _loop(agents, env, episodes=1, progress_bar=False)


def _discretize(obs):
    """
    Convert a continuous 4D CartPole observation into a discrete state tuple.
    
    Parameters:
        obs (array-like): [cart position, cart velocity, pole angle, pole angular velocGity]
    
    Returns:
    """
    return obs


def _loop(agents: list[Agent], env, episodes: int=1000, progress_bar=True) -> dict[str, list[float]]:
    if progress_bar:
        from tqdm import trange
        import logging
        logging.getLogger("root").setLevel(logging.ERROR)
        episode_iterator = trange(1, episodes+1, desc="Training Episodes")
    else:
        episode_iterator = range(1, episodes+1)
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
    if not progress_bar:
        input("Press Enter to continue...")
    return episode_returns
