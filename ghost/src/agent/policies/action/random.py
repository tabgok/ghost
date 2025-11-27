from agent.policy.action_policy import ActionPolicy

class RandomPolicy(ActionPolicy):
    def __init__(self) -> None:
        pass

    def act(self, action_space, observation):
        import random
        return random.choice(action_space)