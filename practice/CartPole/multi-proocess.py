# Run `pip install "gymnasium[classic-control]"` for this example.
import numpy as np
from itertools import repeat
import gymnasium as gym
import cProfile
import pstats
import math
import random
from collections import defaultdict
import multiprocessing
import queue

NUM_BINS = 50  # Recommended: 6–12

pos_bins   = np.linspace(-4.8,   4.8,   NUM_BINS + 1)
vel_bins   = np.linspace(-3.0,   3.0,   NUM_BINS + 1)
angle_bins = np.linspace(-0.418, 0.418, NUM_BINS + 1)
angv_bins  = np.linspace(-3.5,   3.5,   NUM_BINS + 1)

bins = [pos_bins, vel_bins, angle_bins, angv_bins]
LEARNING_QUEUE = None
def init_worker(queue):
    global LEARNING_QUEUE
    LEARNING_QUEUE = queue  # Used to send learnings


def do_final(main_agent):
    env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=100000)
    observation, info = env.reset()
    observation = discretize(observation)
    main_agent.epsilon = 0
    episode_over = False
    total_reward = 0
    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = main_agent.select_action(env.action_space, observation)  # Random action for now - real agents will be smarter!
        prior_observation = observation

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)
        observation = discretize(observation)
        episode_over = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Total reward: {total_reward}")


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
    def __init__(self, state_values = None):
        self.games = 1
        self.state_values = state_values if state_values is not None else defaultdict(float)
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
                if self.state_values.get((observation, action), 0) > max_value:
                    possible_actions = [action]
                    max_value = self.state_values.get((observation, action), 0)
                elif self.state_values.get((observation, action)) == max_value:
                    possible_actions.append(action)
            
            return random.choice(possible_actions)

    def inform(self, prior_state, current_state, action, reward, done):
        if done:
            target = reward
        else:
            max_new = max(
                self.state_values.get((current_state, 0), 0),
                self.state_values.get((current_state, 1), 0),
            )
            target = reward + self.discount_factor * max_new

        self.state_values[(prior_state, action)] = self.state_values.get((prior_state, action), 0) + self.learning_rate*(target - self.state_values.get((prior_state, action), 0))
    
    def end_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

def process_worker(args):
    shared_state = args
    learning_queue = LEARNING_QUEUE
    print("Starting")
    env = gym.make("CartPole-v1")
    agent = RLAgent()
    episodes = 1
    update_count = 1

    while True:
        if episodes % update_count == 0:
            local_dict = dict(shared_state)
            agent.state_values = local_dict
            if local_dict.get("Finished"):
                break

        observation, info = env.reset()
        observation = discretize(observation)
        episode_over = False
        total_reward = 0
        reward = 0
        q = []
        while not episode_over:
            # Choose an action: 0 = push cart left, 1 = push cart right
            action = agent.select_action(env.action_space, observation)  # Random action for now - real agents will be smarter!
            prior_observation = observation

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)
            observation = discretize(observation)
            msg = (prior_observation, observation, action, reward, terminated or truncated)
            agent.inform(*msg)
            #learning_queue.put((prior_observation, observation, action, reward, terminated or truncated))
            q.append(msg)

            total_reward += reward
            episode_over = terminated or truncated
        agent.end_episode()
        learning_queue.put(q)
        learning_queue.put(["episode complete"])
        episodes += 1
    env.close()


def main():
    # Create our training environment - a cart with a pole that needs balancing
    #env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=1000)
    episodes = input("Enter the number of rounds (5 million): ") or 5000000
    episodes = int(episodes)

    manager = multiprocessing.Manager()
    shared_state = manager.dict()  # Used to send updated state/value tables
    #learning_queue = manager.Queue()  # Used to send learnings
    main_agent = RLAgent(state_values=shared_state)
    learning_queue = multiprocessing.Queue()
    init_worker(learning_queue)
    num_processors = multiprocessing.cpu_count()


    with multiprocessing.Pool(processes=num_processors-1, initializer=init_worker, initargs=(learning_queue,)) as pool:
        pool.map_async(process_worker, [shared_state] * (num_processors-1))

        with cProfile.Profile() as pr:
            print("foo")
            while episodes:
                print(f"Remaining: {episodes} - {LEARNING_QUEUE.qsize()}")
                batch = LEARNING_QUEUE.get()
                for msg in batch:
                    if msg == "episode complete":
                        episodes -= 1
                    else:
                        main_agent.inform(*msg)
            print("Sending terminate")
            shared_state["Finished"] = True

            stats = pstats.Stats(pr)
            stats.sort_stats("cumtime").print_stats(40)

    do_final(main_agent)

if __name__ == "__main__":
    main()