from pathlib import Path
from logging import getLogger
import yaml

logger = getLogger(__name__)

BASE_DIR = Path.home() / ".ghost"
AGENTS_DIR = BASE_DIR / "agents"
AGENT_EXT = ".yaml"

ENVIRONMENT_REGISTRY: dict[str, type] = {}


### ENGINE API ###

### --- ENVIRONMENTS --- ###
def list_environments() -> list[str]:
    return list(ENVIRONMENT_REGISTRY.keys())


def register_environment(name: str =None):
    def decorator(cls: type):
        identifier = name or cls.__name__
        ENVIRONMENT_REGISTRY[identifier] = cls
        return cls
    return decorator


### --- TRAINING --- ###
def train(agents: tuple[str], environment: str, episodes: int) -> None:
    logger.info(f"Training agents {agents} in environment {environment} for {episodes} episodes.")

    # Load the environment
    logger.debug(f"Loading environment: {environment}")
    env = ENVIRONMENT_REGISTRY[environment]()



### AGENT CORE ###
class _Core:
    def __init__(self):
        pass

    def _load_agents(self):
        pass

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