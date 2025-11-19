# Run `pip install "gymnasium[classic-control]"` for this example.
import numpy as np
import gymnasium as gym
import math
import random
from collections import defaultdict
import multiprocessing

NUM_BINS = 50  # Recommended: 6–12

pos_bins   = np.linspace(-4.8,   4.8,   NUM_BINS + 1)
vel_bins   = np.linspace(-3.0,   3.0,   NUM_BINS + 1)
angle_bins = np.linspace(-0.418, 0.418, NUM_BINS + 1)
angv_bins  = np.linspace(-3.5,   3.5,   NUM_BINS + 1)

bins = [pos_bins, vel_bins, angle_bins, angv_bins]


def discretize(obs):
    """
    Convert a continuous 4D CartPole observation into a discrete state tuple.
    
    Parameters:
        obs (array-like): [cart position, cart velocity, pole angle, pole angular velocity]
    
    Returns:
        tuple of 4 integers, each in [0, NUM_BINS]
    """
    return tuple(
        int(np.digitize(obs[i], bins[i])) 
        for i in range(4)
    )

class RLAgent:
    def __init__(self):
        self.games = 1
        self.state_values = defaultdict(float)
        self.learning_rate = 0.1  # Do we take the full value, or more slowly converge?
        self.discount_factor = 0.99  # How much do we value the next states vs our current state
        self.min_epsilon = 0.05 # What's our lowest exploration rate?
        self.epsilon = 1 # how often do we explore/just try things?
        self.decay = 0.9999 # How quickly do we drop epsilon?

    def select_action(self, actions, observation):
        if np.random.rand() < self.epsilon:
            return actions.sample()
        else:
            max_value = -1* math.inf
            possible_actions = []
            for action in range(actions.n):
                if self.state_values[(observation, action)] > max_value:
                    possible_actions = [action]
                    max_value = self.state_values[(observation, action )]
                elif self.state_values[(observation, action)] == max_value:
                    possible_actions.append(action)
            
            return random.choice(possible_actions)

    def inform(self, prior_state, current_state, action, reward, done):
        if done:
            target = reward
        else:
            max_new = max(
                self.state_values[(current_state, 0)],
                self.state_values[(current_state, 1)],
            )
            target = reward + self.discount_factor * max_new

        self.state_values[(prior_state, action)] = self.state_values[(prior_state, action)] + self.learning_rate*(target - self.state_values[(prior_state, action)])
    
    def end_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


def main():
    # Create our training environment - a cart with a pole that needs balancing
    #env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=1000)
    env = gym.make("CartPole-v1")
    agent = RLAgent()
    episodes = 5000000
    total_total_reward = 0
    cur_eps = 1

    for episode_count in range(1,episodes+1):
        if episode_count % 10000 == 0:
            env.close()
            env = gym.make("CartPole-v1", render_mode="human")
            cur_eps = agent.epsilon
            agent.epsilon = 0
        # Reset environment to start a new episode
        observation, info = env.reset()
        observation = discretize(observation)

        episode_over = False
        total_reward = 0
        reward = 0
        steps = 0
        while not episode_over:
            # Choose an action: 0 = push cart left, 1 = push cart right
            action = agent.select_action(env.action_space, observation)  # Random action for now - real agents will be smarter!
            prior_observation = observation

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)
            observation = discretize(observation)
            agent.inform(prior_observation, observation, action, reward, terminated or truncated)
            steps += 1
            #print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)

            total_reward += reward
            episode_over = terminated or truncated
        total_total_reward += total_reward

        agent.end_episode()
        if episode_count % 1000 == 0:
            average_total_reward = total_total_reward/1000
            total_total_reward = 0
            print(f"Remaining: {episodes - episode_count}, State Space Size: {len(agent.state_values)}, Average Reward: {average_total_reward}")
        if episode_count % 10000 == 0:
            env.close()
            env = gym.make("CartPole-v1")
            agent.epsilon = cur_eps
            print(f"Evaluation results are: {total_reward}")
        #input()
    env.close()

if __name__ == "__main__":
    main()