from agent.agent import Agent
from agent.agent import _HumanAgent

AGENT_REGISTRY: dict[str, Agent] = {"HumanAgent": _HumanAgent()}

### --- AGENTS --- ###
def register_agent(cls: type):
    AGENT_REGISTRY[cls.__name__] = cls
    return cls


def list_agents() -> list[str]:
    return list(AGENT_REGISTRY.keys())


def describe_agent(agent: str) -> dict[str, dict]:
    details = {}
    if agent not in AGENT_REGISTRY:
        raise NameError(f"Agent '{agent}' is not registered.")

    details = AGENT_REGISTRY[agent].__dict__.copy()
    return details


def create_agent(agent_name: str, **kwargs):
    if agent_name not in AGENT_REGISTRY:
        raise NameError(f"Agent '{agent_name}' is not registered.")
    agent_cls = AGENT_REGISTRY[agent_name]
    return agent_cls(**kwargs)

def rm_agent(agent_name: str) -> None:
    if agent_name not in AGENT_REGISTRY:
        raise NameError(f"Agent '{agent_name}' is not registered.")
    del AGENT_REGISTRY[agent_name]


class _AgentManager:
    def __init__(self):
        pass