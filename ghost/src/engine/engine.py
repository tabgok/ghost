import time
from logging import getLogger


from agent.agent import Agent
from engine import environment_manager
from engine import agent_manager

logger = getLogger(__name__)


### --- TRAINING --- ###
def train(agents: tuple[str], environment: str, episodes: int) -> None:
    logger.info(f"Training agents {agents} in environment {environment} for {episodes} episodes.")

    for agent in agents:
        logger.debug(f"Loading agent: {agent}")
        action_policy = agent_manager._load_agent(agent)
        if action_policy is None:
            logger.warning(f"Agent '{agent}' has no action policy loaded.")
    # Load the environment
    logger.debug(f"Loading environment: {environment}")
    env = environment_manager.instantiate_environment(environment)


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
        obs (array-like): [cart position, cart velocity, pole angle, pole angular velocity]
    
    Returns:
    """
    return obs


def _loop(agents: list[Agent], env, episodes: int=1000, progress_bar=True) -> None:
    for _ in range(1, episodes+1):
        # Reset environment to start a new episode
        observation, info = env.reset()
        observation = _discretize(observation)

        episode_over = False
        total_reward = 0
        reward = 0
        steps = 0
        turn = 0
        while not episode_over:
            agent = agents[turn]
            turn = (turn + 1) % len(agents)
            # Choose an action: 0 = push cart left, 1 = push cart right
            action = agent.select_action(env.action_space, env.observation_space, observation)  # Random action for now - real agents will be smarter!
            prior_observation = observation

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)
            observation = _discretize(observation)
            agent.learn(prior_observation, observation, action, reward, terminated or truncated)
            steps += 1
            #print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)

            total_reward += reward
            episode_over = terminated or truncated

        agent.end_episode()
    env.close()
    time.sleep(1)