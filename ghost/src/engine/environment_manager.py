import gymnasium as gym


ENVIRONMENT_REGISTRY: dict[str, type] = {}

### --- ENVIRONMENTS --- ###
def list_environments() -> list[str]:
    return list(ENVIRONMENT_REGISTRY.keys())


def describe_environment(env_name: str) -> dict[str, dict]:
    details = {}
    if env_name not in ENVIRONMENT_REGISTRY:
        raise NameError(f"Environment '{env_name}' is not registered.")

    details = ENVIRONMENT_REGISTRY[env_name].__dict__.copy()
    return details

def register_environment(name: str =None):
    def decorator(cls: type):
        identifier = name or cls.__name__
        ENVIRONMENT_REGISTRY[identifier] = cls
        return cls
    return decorator

def instantiate_environment(env_name: str, **kwargs):
    if env_name not in ENVIRONMENT_REGISTRY:
        raise NameError(f"Environment '{env_name}' is not registered.")
    return gym.make(env_name, render_mode=kwargs.get("render_mode", None))