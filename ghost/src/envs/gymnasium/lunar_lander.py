class LunarLanderEnv:
    """Placeholder wrapper for Gymnasium LunarLander-v2."""

    env_id = "LunarLander-v2"

    def reset(self):
        raise NotImplementedError("LunarLanderEnv.reset is not implemented yet.")

    def step(self, action):
        raise NotImplementedError("LunarLanderEnv.step is not implemented yet.")
