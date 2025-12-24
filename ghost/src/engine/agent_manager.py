from agent.policies.action.policy import ACTION_POLICY_REGISTRY 

from agent.policies.exploration.policy import EXPLORATION_POLICY_REGISTRY
from agent.policies.learning.policy import LEARNING_POLICY_REGISTRY
from agent.agent import Agent


POLICY_REGISTRY: dict[str, Agent] = {}
AGENT_REGISTRY: dict[str, Agent] = {}


### --- AGENTS --- ###
def list_agents() -> list[str]:
    return list(AGENT_REGISTRY.keys())


def register_agent(agent: Agent) -> None:
    AGENT_REGISTRY[agent.__name__] = agent


@register_agent
class HumanAgent(Agent):
    def __init__(self, name: str = "HumanAgent") -> None:
        super().__init__(
            name="HumanAgent",
            learning_policy=LEARNING_POLICY_REGISTRY["NoOpLearningPolicy"],
            action_policy=ACTION_POLICY_REGISTRY["HumanActionPolicy"],
            exploration_policy=EXPLORATION_POLICY_REGISTRY["NoOpExplorationPolicy"],
        )


@register_agent
class RandomAgent(Agent):
    def __init__(self, name: str = "RandomAgent") -> None:
        super().__init__(
            name="RandomAgent",
            learning_policy=LEARNING_POLICY_REGISTRY["NoOpLearningPolicy"],
            action_policy=ACTION_POLICY_REGISTRY["RandomActionPolicy"],
            exploration_policy=EXPLORATION_POLICY_REGISTRY["NoOpExplorationPolicy"],
        )


@register_agent
class GreedyAgent(Agent):
    def __init__(self, name: str = "GreedyAgent") -> None:
        super().__init__(
            name="GreedyAgent",
            learning_policy=LEARNING_POLICY_REGISTRY["NoOpLearningPolicy"],
            action_policy=ACTION_POLICY_REGISTRY["GreedyPolicy"],
            exploration_policy=EXPLORATION_POLICY_REGISTRY["NoOpExplorationPolicy"],
        )


@register_agent
class MonteCarloAgent(Agent):
    def __init__(self, name: str = "MonteCarloAgent", save_file: str = None) -> None:
        super().__init__(
            name="MonteCarloAgent",
            learning_policy=LEARNING_POLICY_REGISTRY["MonteCarloLearningPolicy"],
            action_policy=ACTION_POLICY_REGISTRY["GreedyPolicy"],
            exploration_policy=EXPLORATION_POLICY_REGISTRY["EpsilonDecayExplorationPolicy"],
        )