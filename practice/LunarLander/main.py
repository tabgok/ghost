import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import tqdm
import cProfile
import pstats
import numba as nb

class RandomAgent:
    def __init__(self, observation_space=None):
        pass
    
    def select_action(self, action_space, observation):
        return action_space.sample()

    def update(self, reward):
        pass

rng = np.random.default_rng()

@nb.njit
def discretize_jit(obs, edges):
    out = np.empty(obs.shape[0], dtype=np.int64)
    for i in range(obs.shape[0]):
        out[i] = np.searchsorted(edges[i], obs[i], side='right')
    return out

class QAgent():
    def __init__(self, observation_space=None, action_space=None):
        self.state_action = defaultdict(float)
        self.step_size = 0.1  # How quickly we learn (how much impact the reward has on update, 1 is full replacement)
        self.exploration_rate = 0.8  # How much we explore (initially)
        self.exploration_rate_min = 0.01  # Minimum exploration rate
        self.exploration_rate_decay = 0.999  # How quickly we lower the exploration rate
        self.discount_factor = 0.99   # How much we care about future values
        self.observation_space = observation_space
        self.action_space = action_space
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
        last_state_space = self.discretize(last_observation)
        current_state_space = self.discretize(current_observation)
        state_action_space = (last_state_space, last_action)
        # last = last + learning_rate * (reward + discount_factor*max_current - last)
        last = self.state_action[state_action_space]
        max_current = 0.0 if done else max(self.state_action[(current_state_space, i)] for i in range(self.action_space.n))
        self.state_action[state_action_space] = last + self.step_size*(reward + self.discount_factor*max_current - last)

    def _make_buckets(self, n_bins=10):
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


def main():
    episodes = 10000000
    display_count = 0

    render_mode = None
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
            enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode=render_mode)
    rewards = []
    agent = QAgent(env.observation_space, env.action_space)
    last_exploration_rate = 0
    bucket_reward = 0

    #with cProfile.Profile() as pr:
    for episode in tqdm.trange(episodes, desc="Episodes"):
        episode_reward = 0
        if display_count and (episode % (episodes//display_count) == 0):
            render_mode = "human"
            bucket_reward = 0
            last_exploration_rate = agent.exploration_rate
            agent.exploration_rate = 0.0
        else:
            render_mode = None
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode=render_mode)

        last_observation, info = env.reset()

        episode_over = False

        while not episode_over:
            last_action = agent.select_action(env.action_space, last_observation)
            current_observation, reward, terminated, truncated, info = env.step(last_action)
            episode_over = terminated or truncated
            episode_reward += reward
            agent.update(reward, last_observation, last_action, current_observation, episode_over)
            last_observation = current_observation
        if display_count and (episode % (episodes//10) == 0):
            agent.exploration_rate = last_exploration_rate
            print(f"Evaluation run reward: {episode_reward}, final reward: {reward}, agent memsize: {len(agent.state_action.keys())}, exploration rate: {agent.exploration_rate}")
        rewards.append(episode_reward)
        #print(episode, episode_reward)
        bucket_reward += episode_reward
        env.close()
        #stats = pstats.Stats(pr)
        #stats.sort_stats("cumtime").print_stats(40)
    
    display_learning(rewards)


def display_learning(rewards):
    rewards_array = np.array(rewards)
    # Plot the rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_array)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot rewards with moving average
    plt.subplot(1, 2, 2)
    window_size = 100
    moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.title(f'Moving Average of Rewards (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
