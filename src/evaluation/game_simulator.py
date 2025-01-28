from constants import *
from enviroment.game_board import GameBoard
import time
import matplotlib.pyplot as plt
import psutil
import os
import math

class GameSimulator:
    def __init__(self, agent1, agent1_symbol, agent2, agent2_symbol):
        self.agent1 = agent1
        self.agent1_symbol = agent1_symbol
        self.agent2 = agent2
        self.agent2_symbol = agent2_symbol
        
        self.reset_stats()
    
    
    def reset_stats(self):
        self.agent1_wins = 0
        self.agent2_wins = 0
        self.draws = 0
        self.game_lengths = []
        self.agent1_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
        self.agent2_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
        self.execution_times = []
        self.memory_usages = []
    
    
    def simulate(self, num_games):
        self.reset_stats()
        
        for game in range(num_games):
            start_time = time.time()
            board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
            agents = {
                self.agent1_symbol: self.agent1,
                self.agent2_symbol: self.agent2
            }
            
            current_player = self.agent1_symbol
            moves = 0
            
            while True:
                moves += 1
                # Make move
                move = agents[current_player].get_move(board.board)
                board.insert_token(move, current_player)
                
                # Check for winner
                if board.check_winner(current_player):
                    if current_player == self.agent1_symbol:
                        self.agent1_wins += 1
                        pattern = board.get_winning_pattern(current_player)
                        self.agent1_patterns[pattern] += 1
                    else:
                        self.agent2_wins += 1
                        pattern = board.get_winning_pattern(current_player)
                        self.agent2_patterns[pattern] += 1
                    break
                
                # Check for draw
                if board.is_draw():
                    self.draws += 1
                    break
                
                # Switch player
                current_player = self.agent2_symbol if current_player == self.agent1_symbol else self.agent1_symbol

            # Record game metrics
            self.game_lengths.append(moves)
            self.execution_times.append(time.time() - start_time)
            
            # Measure memory usage
            process = psutil.Process(os.getpid())
            self.memory_usages.append(process.memory_info().rss / (1024 * 1024))

        self._print_results(num_games)
        self._plot_results(num_games)
    
    def _print_results(self, num_games):
        print(f"\nResults after {num_games} games:")
        print(f"{self.agent1.__class__.__name__} ({self.agent1_symbol}) wins: {self.agent1_wins}")
        print(f"{self.agent2.__class__.__name__} ({self.agent2_symbol}) wins: {self.agent2_wins}")
        print(f"Draws: {self.draws}")

        # Accuracy metrics
        print("\nAccuracy Metrics:")
        print(f"{self.agent1.__class__.__name__} Win Rate: {self.agent1_wins/num_games:.2f}")
        print(f"{self.agent2.__class__.__name__} Win Rate: {self.agent2_wins/num_games:.2f}")
        print(f"Draw Rate: {self.draws/num_games:.2f}")

        # Efficiency metrics
        print("\nEfficiency Metrics:")
        print(f"Avg. Execution Time: {sum(self.execution_times)/num_games:.2f}s")
        print(f"Avg. Memory Usage: {sum(self.memory_usages)/num_games:.2f}MB")
        print(f"Max Memory Usage: {max(self.memory_usages):.2f}MB")

        # Game metrics
        print("\nGame Metrics:")
        print(f"Avg. Game Length: {sum(self.game_lengths)/num_games:.2f} moves")
        print(f"{self.agent1.__class__.__name__} Patterns: {self.agent1_patterns}")
        print(f"{self.agent2.__class__.__name__} Patterns: {self.agent2_patterns}")
    
    
    def _plot_results(self, num_games):
        plt.figure(figsize=(18, 12))
        
        # Win rates
        plt.subplot(2, 3, 1)
        labels = [
            f"{self.agent1.__class__.__name__}",
            f"{self.agent2.__class__.__name__}",
            "Draws"
        ]
        plt.bar(labels, [self.agent1_wins, self.agent2_wins, self.draws], color=['blue', 'orange', 'green'])
        plt.title('Win Rates')
        plt.ylabel('Number of Games')

        # Game length distribution
        plt.subplot(2, 3, 2)
        plt.hist(self.game_lengths, bins=range(min(self.game_lengths), max(self.game_lengths)+1), alpha=0.75, color='purple')
        plt.title('Game Length Distribution')
        plt.xlabel('Moves')
        plt.ylabel('Frequency')

        # Winning patterns
        plt.subplot(2, 3, 3)
        patterns = ['horizontal', 'vertical', 'diagonal']
        a1_counts = [self.agent1_patterns[p] for p in patterns]
        a2_counts = [self.agent2_patterns[p] for p in patterns]
        
        bar_width = 0.35
        index = range(len(patterns))
        
        plt.bar(index, a1_counts, bar_width, color='blue', label=self.agent1.__class__.__name__)
        plt.bar([i + bar_width for i in index], a2_counts, bar_width, color='orange', label=self.agent2.__class__.__name__)
        plt.xticks([i + bar_width/2 for i in index], patterns)
        plt.title('Winning Patterns')
        plt.xlabel('Pattern Type')
        plt.ylabel('Frequency')
        plt.legend()

        # Memory usage
        plt.subplot(2, 3, 4)
        plt.plot(range(1, num_games+1), self.memory_usages, marker='o', color='red')
        plt.title('Memory Usage')
        plt.xlabel('Game')
        plt.ylabel('MB')
        plt.grid(True)

        # Execution time
        plt.subplot(2, 3, 5)
        plt.plot(range(1, num_games+1), self.execution_times, marker='o', color='green')
        plt.title('Execution Time')
        plt.xlabel('Game')
        plt.ylabel('Seconds')
        plt.grid(True)

        plt.tight_layout()
        plt.show()