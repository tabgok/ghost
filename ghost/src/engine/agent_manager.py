import yaml
from pathlib import Path

from agent.agent import Agent
from agent.agent import _HumanAgent


BASE_DIR = Path.home() / ".ghost"
AGENT_REGISTRY: dict[str, Agent] = {"HumanAgent": _HumanAgent}
AGENTS_DIR = BASE_DIR / "agents"
AGENT_EXT = ".yaml"

### --- AGENTS --- ###
def load_agent(agent_name: str):
    if agent_name not in AGENT_REGISTRY:
        raise NameError(f"Agent '{agent_name}' is not registered.")
    agent_cls = AGENT_REGISTRY[agent_name]
    return agent_cls()

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


### UTILITY FUNCTIONS ###
def read_agent_config(agent_name: str) -> dict:
    agent_path = AGENTS_DIR / f"{agent_name}{AGENT_EXT}"
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent '{agent_name}' not found at {agent_path}")
    with agent_path.open(encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


        return []
    return sorted(p.stem for p in AGENTS_DIR.glob(f"*{AGENT_EXT}") if p.is_file())


def load_action_policy_from_file(agent_name: str):
    if agent_name.lower() == "human":
        # TODO: Return policies
        return 
    agent_cfg = read_agent_config(agent_name)
    # TODO: Return policiesreturn load_action_policy(agent_cfg)
    return None