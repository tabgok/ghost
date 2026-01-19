from collections import defaultdict
import math
import numba as nb
import numpy as np
from pathlib import Path
import pickle

rng = np.random.default_rng()

class TabularQAgent():
    def __init__(self, observation_space=None, action_space=None):
        self.state_action = defaultdict(float)
        self.step_size = 0.1  # How quickly we learn (how much impact the reward has on update, 1 is full replacement)
        self.exploration_rate = 0.8  # How much we explore (initially)
        self.exploration_rate_min = 0.01  # Minimum exploration rate
        self.exploration_rate_decay = 0.999  # How quickly we lower the exploration rate
        self.discount_factor = 0.99   # How much we care about future values
        self.observation_space = observation_space
        self.action_space = action_space
        self.updates = 0
        self._make_buckets()

    def select_action(self, action_space, observation):
        if rng.random() < self.exploration_rate:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
            return action_space.sample()
        else:
            state_space = self.discretize(observation)
            m = -1*math.inf
            action = -1
            for i in range(action_space.n):
                val = self.state_action[(state_space, i)]
                if val > m:
                    m = val
                    action = i
            return action
    
    def update(self, reward, last_observation, last_action, current_observation, done):
        self.updates += 1
        last_state_space = self.discretize(last_observation)
        current_state_space = self.discretize(current_observation)
        state_action_space = (last_state_space, last_action)
        # last = last + learning_rate * (reward + discount_factor*max_current - last)
        last = self.state_action[state_action_space]
        max_current = 0.0 if done else max(self.state_action[(current_state_space, i)] for i in range(self.action_space.n))
        self.state_action[state_action_space] = last + self.step_size*(reward + self.discount_factor*max_current - last)

    def _make_buckets(self, n_bins=40):
        self.dimensions = self.observation_space.shape[0]
        bins_per_dimension = n_bins#int(math.pow(n_bins, 1./self.dimensions))
        self.bins = []
        for dimension in range(self.dimensions):
            low = self.observation_space.low[dimension]
            high = self.observation_space.high[dimension]
            # linspace turns a range into bins
            self.bins.append(np.linspace(low, high, num=bins_per_dimension))
        self.bin_edges = np.asarray(self.bins)
            
    def discretize(self, observation):
        # Digitize turns values into bin indices
        #obs_arr = np.asarray(observation)
        #5return tuple(np.digitize(observation[i], self.bins[i]) for i in range(self.dimensions))
        obs_arr = np.asarray(observation)
        # vectorized search: still loops in Python over dims but avoids generator overhead
        #return tuple(np.searchsorted(self.bin_edges[i], obs_arr[i], side='right') for i in range(self.dimensions))
        return tuple(discretize_jit(obs_arr, self.bin_edges))

    def save(self, path: Path):
        data = {
            "state_action": dict(self.state_action),
            "exploration_rate": self.exploration_rate,
            "step_size": self.step_size,
            "exploration_rate_min": self.exploration_rate_min,
            "exploration_rate_decay": self.exploration_rate_decay,
            "discount_factor": self.discount_factor,
            "bins": self.bin_edges,
        }
        with path.open("wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path):
        if not path.exists():
            return
        with path.open("rb") as f:
            data = pickle.load(f)
        self.state_action = defaultdict(float, data.get("state_action", {}))
        self.exploration_rate = data.get("exploration_rate", self.exploration_rate)
        self.step_size = data.get("step_size", self.step_size)
        self.exploration_rate_min = data.get("exploration_rate_min", self.exploration_rate_min)
        self.exploration_rate_decay = data.get("exploration_rate_decay", self.exploration_rate_decay)
        self.discount_factor = data.get("discount_factor", self.discount_factor)
        bins_loaded = data.get("bins")
        if bins_loaded is not None:
            self.bin_edges = bins_loaded

@nb.njit
def discretize_jit(obs, edges):
    out = np.empty(obs.shape[0], dtype=np.int64)
    for i in range(obs.shape[0]):
        out[i] = np.searchsorted(edges[i], obs[i], side='right')
    return out

class RandomAgent:
    def __init__(self, observation_space=None):
        pass
    
    def select_action(self, action_space, observation):
        return action_space.sample()

    def update(self, reward):
        pass