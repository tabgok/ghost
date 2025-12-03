from .engine import (
    train,
    evaluate
)

from .agent_manager import (
    create_agent,
    describe_agent,
    rm_agent as delete_agent,
    register_agent,
    list_agents,
)


from .environment_manager import (
    describe_environment,
    list_environments,
    register_environment
)