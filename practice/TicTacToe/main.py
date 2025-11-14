import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm.contrib.concurrent import thread_map  # or thread_map
import multiprocessing as mp
from collections import defaultdict
"""
We need:
* State
* Action
* Reward
"""
class RandomPlayer:
    def choose_action(self, possible_actions, board, player_value):
        return random.choice(possible_actions) 
    
    def inform(self, game_sequence, game_result):
        pass

class RLPlayer:
    def __init__(self):
        self.state_values = defaultdict(int)
        self.discount_factor = 0.9
        self.experimentation_rate = 0.8
        self.games = 0

    def choose_action(self, possible_actions, board, player_value):
        if np.random.uniform(0, 1) <= self.experimentation_rate * 1/(self.games+1):
            return random.choice(possible_actions)
        
        value_max = -1*math.inf
        final_actions = []
        for action in possible_actions:
            board_copy = board.copy()
            board_copy.move(action, player_value)
            hash = board_copy.hash()
            if self.state_values[hash] > value_max:
                value_max = self.state_values[hash]
                final_actions = [action]
            elif self.state_values[hash] == value_max:
                final_actions.append(action)
        return random.choice(final_actions)

    def inform(self, game_sequence, reward):
        self.games += 1
        for state in reversed(game_sequence):
            self.state_values[state] = self.state_values[state] + self.discount_factor*(reward - self.state_values[state])
            reward = self.state_values[state]


class HumanPlayer:
    def choose_action(self, possible_actions, board, player_value):
        while True:
            try:
                print("Enter your move as row and column (0-2 each):")
                row = int(input("Row: "))
                col = int(input("Column: "))
                if (row, col) in possible_actions:
                    return (row, col)
                else:
                    print("Invalid move! Try again.")
            except ValueError:
                print("Please enter valid numbers between 0 and 2.")

    def inform(self, game_sequence, game_result):
        pass

class Game:
    def __init__(self, p1=None, p2=None, display=False):
        if not p1:
            p1 = RLPlayer()
        if not p2:
            p2 = RandomPlayer()

        self.players = [p1, p2]
        self.current_player_idx = 0
        self.board = Board()
        self.display = display
        self.states = []
    
    def play(self):
        moves = self.board.valid_moves()
        while moves and not self.check_win():
            if self.display:
                print(self.board)
                print()
            player_value = -1 if not self.current_player_idx else 1
            action = self.players[self.current_player_idx].choose_action(moves, self.board, player_value)
            self.board.move(action, player_value)
            self.states.append(self.board.hash())
            self.current_player_idx = (self.current_player_idx + 1) % 2
            moves = self.board.valid_moves()

        if self.display:
            print(self.board)
            if self.check_win():
                print("Victory for player: ", 'X' if self.check_win() == -1 else 'O')
            else:
                print("The game is a draw!")
        return self.check_win()


    def check_win(self):
        # Check rows
        board = self.board.state
        for row in board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]
        
        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != 0:
                return board[0][col]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != 0:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != 0:
            return board[0][2]
        
        # Check for tie
        for row in board:
            if 0 in row:
                return None  # Game not finished yet
        
        return 0  # Tie



class Board:
    def __init__(self):
        self.state = np.zeros((3,3))

    def copy(self):
        board = Board()
        board.state = self.state.copy()
        return board
    
    def valid_moves(self):
        moves = []
        for r in range(3):
            for c in range(3):
                if self.state[r][c] == 0:
                    moves.append((r,c))
        
        return moves

    def move(self, coords, val):
        r, c = coords
        self.state[r][c] = val
    
    def hash(self):
        return str(self.state.reshape(3 * 3))

    def __str__(self):
        result = ""
        for i, row in enumerate(self.state):
            for j, cell in enumerate(row):
                if cell == -1:
                    result += "X"
                elif cell == 1:
                    result += "O"
                else:
                    result += " "
                if j < 2:
                    result += "|"
            if i < 2:
                result += "\n-----\n"
        return result


    

def display(p1, p1_results, p2, p2_results):
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'round': range(rounds),
        'p1_wins': p1_results,
        'p2_wins': p2_results
    })
    
    # Calculate moving averages
    window_size = max(1, rounds // 32)
    df['p1_moving_avg'] = df['p1_wins'].rolling(window=window_size, min_periods=1).mean()
    df['p2_moving_avg'] = df['p2_wins'].rolling(window=window_size, min_periods=1).mean()
    
    # Display statistics
    p1_wins = sum(p1_results)
    p2_wins = sum(p2_results)
    print(f"Total games: {len(p1_results)}")
    print(f"Player 1 ({type(p1).__name__}) (X) win rate: {p1_wins / rounds:.2%}")
    print(f"Player 2 ({type(p2).__name__}) (O) win rate: {p2_wins / rounds:.2%}")
    
    # Create visualization
    plt.switch_backend('TkAgg')
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Win rates over time
    plt.subplot(1, 2, 1)
    plt.plot(df['round'], df['p1_moving_avg'], label=f'Player 1 ({type(p1).__name__}) (X)', linewidth=2)
    plt.plot(df['round'], df['p2_moving_avg'], label=f'Player 2 ({type(p2).__name__}) (O)', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Win Rate')
    plt.title('Tic Tac Toe Game Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative wins
    plt.subplot(1, 2, 2)
    p1_cumulative = df['p1_wins'].cumsum()
    p2_cumulative = df['p2_wins'].cumsum()
    plt.plot(p1_cumulative, label=f'Player 1 ({type(p1).__name__}) (X)', linewidth=2)
    plt.plot(p2_cumulative, label=f'Player 2 ({type(p2).__name__}) (O)', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Cumulative Wins')
    plt.title('Cumulative Wins Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print bucketed results
    bucket_size = max(1, rounds // 50)
    print(f"\nBucketed Results (bucket size: {bucket_size}):")
    for i in range(0, rounds, bucket_size):
        end = min(i + bucket_size, rounds)
        p1_bucket = sum(p1_results[i:end])
        p2_bucket = sum(p2_results[i:end])
        p1_rate = p1_bucket / bucket_size
        p2_rate = p2_bucket / bucket_size
        print(f"Bucket {i//bucket_size + 1}: P1 = {p1_rate:.2%}, P2 = {p2_rate:.2%}")

def play_game_chunk(args):
    """Play a chunk of games and return results"""
    start_round, end_round, p1, p2 = args
    p1_results = []
    p2_results = []
    
    for _ in range(start_round, end_round):
        game = Game(p1, p2)
        result = game.play()
        if result == -1:
            p1.inform(game.states, 1)
            p2.inform(game.states, -1)
        elif result == 0:
            p1.inform(game.states, 0)
            p2.inform(game.states, 0)
        else:
            p1.inform(game.states, -1)
            p2.inform(game.states, 1)

        p1_results.append(1 if result == -1 else 0)
        p2_results.append(1 if result == 1 else 0)
    
    return p1_results, p2_results


def run_single_threaded(rounds):
    p1_results = []
    p2_results = []
    for _ in tqdm.tqdm(range(rounds), desc="playing games"):
        game = Game(p1, p2)
        p1_chunk, p2_chunk = play_game_chunk((0, 1, p1, p2))
        p1_results.extend(p1_chunk)
        p2_results.extend(p2_chunk)

    return p1_results, p2_results

def run_multi_threaded(rounds):
    # Use multithreading to split rounds across threads
    num_threads = 32
    rounds_per_thread = rounds // num_threads
    remaining_rounds = rounds % num_threads
    
    # Create arguments for each thread
    thread_args = []
    start = 0
    
    for i in range(num_threads):
        end = start + rounds_per_thread + (1 if i < remaining_rounds else 0)
        thread_args.append((start, end, p1, p2))
        start = end
    
    # Play games in parallel with progress bar
    results = thread_map(play_game_chunk, thread_args, desc="Processing", chunksize=1, max_workers=num_threads)
    
    # Aggregate results
    p1_results = []
    p2_results = []
    for p1_chunk, p2_chunk in results:
        p1_results.extend(p1_chunk)
        p2_results.extend(p2_chunk)

    return p1_results, p2_results


def run_multi_process(rounds):
    # Use multiprocessing to split rounds across processors
    num_processes = mp.cpu_count()
    rounds_per_process = rounds // num_processes
    remaining_rounds = rounds % num_processes
    
    # Create arguments for each process
    process_args = []
    start = 0
    
    for i in range(num_processes):
        end = start + rounds_per_process + (1 if i < remaining_rounds else 0)
        process_args.append((start, end, p1, p2))
        start = end
    
    # Play games in parallel with progress bar
    results = process_map(play_game_chunk, process_args, chunksize=1, desc="Processing", max_workers=num_processes)
    
    # Aggregate results
    p1_results = []
    p2_results = []
    for p1_chunk, p2_chunk in results:
        p1_results.extend(p1_chunk)
        p2_results.extend(p2_chunk)

    return p1_results, p2_results


if __name__ == '__main__':
    print("Choose player types:")
    print("1. Human")
    print("2. Random")
    print("3. RL")
    
    choice1 = input("Enter choice for Player 1 (1-3): ")
    choice2 = input("Enter choice for Player 2 (1-3): ")
    
    player_types = [HumanPlayer, RandomPlayer, RLPlayer]
    
    if choice1 == "1":
        p1 = HumanPlayer()
    elif choice1 == "2":
        p1 = RandomPlayer()
    elif choice1 == "3":
        p1 = RLPlayer()
    else:
        print("Invalid choice for Player 1, using Random")
        p1 = RandomPlayer()
    
    if choice2 == "1":
        p2 = HumanPlayer()
    elif choice2 == "2":
        p2 = RandomPlayer()
    elif choice2 == "3":
        p2 = RLPlayer()
    else:
        print("Invalid choice for Player 2, using Random")
        p2 = RandomPlayer()
    
    rounds = int(input("How many rounds? "))

    mechanism = int(input("1 for single-thread, 2 for multi-thread, 3 for multi-process: "))

    execution_map = {
        1: run_single_threaded,
        2: run_multi_threaded,
        3: run_multi_process
    }
    
    p1_results, p2_results = execution_map[mechanism](rounds)

    display(p1, p1_results, p2, p2_results)