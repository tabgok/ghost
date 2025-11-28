class CartPoleEnv:
    """Placeholder wrapper for Gymnasium CartPole-v1."""

    env_id = "CartPole-v1"

    def reset(self):
        raise NotImplementedError("CartPoleEnv.reset is not implemented yet.")

    def step(self, action):
        raise NotImplementedError("CartPoleEnv.step is not implemented yet.")
