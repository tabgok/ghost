import yaml
from pathlib import Path

from agent.policies.action.policy import ACTION_POLICY_REGISTRY 
from agent.policies.exploration.policy import EXPLORATION_POLICY_REGISTRY
from agent.policies.learning.policy import LEARNING_POLICY_REGISTRY
from agent.agent import Agent


BASE_DIR = Path.home() / ".ghost"
POLICY_REGISTRY: dict[str, Agent] = {}
AGENT_REGISTRY: dict[str, Agent] = {}
AGENTS_DIR = BASE_DIR / "agents"
AGENT_EXT = ".yaml"


### --- AGENTS --- ###
def list_agents() -> list[str]:
    return list(AGENT_REGISTRY.keys())


def describe_agent(agent_name: str) -> dict[str, dict]:
    details = {}
    if agent_name not in AGENT_REGISTRY:
        raise NameError(f"Agent '{agent_name}' is not registered.")

    details = AGENT_REGISTRY[agent_name].snapshot()
    return details


def rm_agent(agent_name: str) -> None:
    if agent_name not in AGENT_REGISTRY:
        raise NameError(f"Agent '{agent_name}' is not registered.")
    del AGENT_REGISTRY[agent_name]


def create_agent(**kwargs) -> str:
    agent_name = kwargs.get("name")
    action_policy = ACTION_POLICY_REGISTRY[kwargs.get("action_policy")]
    exploration_policy = EXPLORATION_POLICY_REGISTRY[kwargs.get("exploration_policy")]
    learning_policy = LEARNING_POLICY_REGISTRY[kwargs.get("learning_policy")]
    agent_instance = Agent(name=agent_name,
                           action_policy=action_policy,
                           exploration_policy=exploration_policy,
                           learning_policy=learning_policy)
    AGENT_REGISTRY[agent_name] = agent_instance
    agent_instance.snapshot()
    _save_agent_to_disk(agent_name)
    return agent_name

### --- POLICIES --- ###
def list_action_policies() -> list[str]:
    return list(ACTION_POLICY_REGISTRY.keys())

def list_exploration_policies() -> list[str]:
    return list(EXPLORATION_POLICY_REGISTRY.keys())

def list_learning_policies() -> list[str]:
    return list(LEARNING_POLICY_REGISTRY.keys())




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


def _load_agent(agent_name: str, **kwargs) -> Agent:

    if agent_name not in AGENT_REGISTRY:
        instance = Agent.from_snapshot(kwargs)
        AGENT_REGISTRY[agent_name] = instance
    instance = AGENT_REGISTRY[agent_name]
    return instance


def _load_agents_from_disk() -> None:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    for agent_file in AGENTS_DIR.glob(f"*{AGENT_EXT}"):
        agent_name = agent_file.stem
        if agent_name not in AGENT_REGISTRY:
            agent_cfg = read_agent_config(agent_name)
            agent_instance = _load_agent(agent_name, **agent_cfg)
            AGENT_REGISTRY[agent_name] = agent_instance


def _save_agent_to_disk(agent_name: str) -> None:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    agent_path = AGENTS_DIR / f"{agent_name}{AGENT_EXT}"
    agent_instance = AGENT_REGISTRY.get(agent_name)
    if not agent_instance:
        raise NameError(f"Agent '{agent_name}' is not registered.")
    with agent_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(agent_instance.snapshot(), fp)


# Load agents from disk at module import time
_load_agents_from_disk()