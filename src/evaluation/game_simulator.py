from constants import *
from enviroment.game_board import GameBoard
import time
import matplotlib.pyplot as plt
import psutil
import os
import math

class GameSimulator:
    def __init__(self, agent1, agent1_symbol, agent2, agent2_symbol):
        # creates the attributes [agent1] & [agent2] that safes a Object from a Agent class
        self.agent1 = agent1 
        self.agent2 = agent2
        
        # creates attributes [agent1_symbol] & [agent2_symbol] that assigns a symbol to a agent
        self.agent1_symbol = agent1_symbol
        self.agent2_symbol = agent2_symbol
        
        # create measurements for the tests
        self.reset_stats()
    
    
    def reset_stats(self):
        '''
        the method [reset_stats] can be used to either create measurements to messure overwach
        the game simulation or to reset the created measurements.
        The following measurements will be created:
         ○ agent1 wins
         ○ agent2 wins
         ○ draws
         ○ game lenght
         ○ winning patterns
         ○ execution time
         ○ memory usage
        '''
        
        self.agent1_wins = 0
        self.agent2_wins = 0
        self.draws = 0
        self.game_lengths = []
        self.agent1_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
        self.agent2_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
        self.execution_times = []
        self.memory_usages = []
    
    
    def simulate(self, num_games):
        '''
        the method [simulate] can be used to simulate games between two agents and overwatch some 
        messurements.
        The paramester [num_games] can be used to specify how many games the method should simulate.
        '''
        
        # reset all messurements
        self.reset_stats()
        
        for game in range(num_games):
            
            # start timer for a game to keep track
            start_time = time.time()
            
            # create a new gameboard
            board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
            
            # assign aymbols to agents
            agents = {
                self.agent1_symbol: self.agent1,
                self.agent2_symbol: self.agent2
            }
            
            # specify who is the current player
            current_player = self.agent1_symbol
            
            # keep track of the moves
            moves = 0
            
            # play game
            while True:
                moves += 1
                
                # get new move from current agent
                move = agents[current_player].get_move(board.board)
                
                # intert move into the board
                board.insert_token(move, current_player)
                
                # check if the current player won after the played move
                if board.check_winner(current_player):
                    if current_player == self.agent1_symbol:
                        
                        # keep track of messurements
                        self.agent1_wins += 1
                        pattern = board.get_winning_pattern(current_player)
                        self.agent1_patterns[pattern] += 1
                    else:
                        
                        # keep track of messurements
                        self.agent2_wins += 1
                        pattern = board.get_winning_pattern(current_player)
                        self.agent2_patterns[pattern] += 1
                    break
                
                # check if board is a draw
                if board.is_draw():
                    self.draws += 1
                    break
                
                # change current player to other agent
                current_player = self.agent2_symbol if current_player == self.agent1_symbol else self.agent1_symbol

            # keep track of messurements
            self.game_lengths.append(moves)
            self.execution_times.append(time.time() - start_time)
            
            process = psutil.Process(os.getpid())
            self.memory_usages.append(process.memory_info().rss / (1024 * 1024))

        # print result of game simmulation
        self._print_results(num_games)
        
        # plot result of game simmulation
        self._plot_results(num_games)
    
    def _print_results(self, num_games):
        '''
        the privat method [_print_results] can be used to print the result of the messuremnts of one
        game simmulation.
        The parameter [num_games] is the amount of games the simulator simulated
        '''
        
        # amount of wins and draws
        print(f"\nResults after {num_games} games:")
        print(f"{self.agent1.__class__.__name__} ({self.agent1_symbol}) wins: {self.agent1_wins}")
        print(f"{self.agent2.__class__.__name__} ({self.agent2_symbol}) wins: {self.agent2_wins}")
        print(f"Draws: {self.draws}")

        # winrate and drawrate of the agents
        print("\nAccuracy Metrics:")
        print(f"{self.agent1.__class__.__name__} Win Rate: {self.agent1_wins/num_games:.2f}")
        print(f"{self.agent2.__class__.__name__} Win Rate: {self.agent2_wins/num_games:.2f}")
        print(f"Draw Rate: {self.draws/num_games:.2f}")

        # execution time and memor usage
        print("\nEfficiency Metrics:")
        print(f"Avg. Execution Time: {sum(self.execution_times)/num_games:.2f}s")
        print(f"Avg. Memory Usage: {sum(self.memory_usages)/num_games:.2f}MB")
        print(f"Max Memory Usage: {max(self.memory_usages):.2f}MB")

        # winning patterns
        print("\nGame Metrics:")
        print(f"Avg. Game Length: {sum(self.game_lengths)/num_games:.2f} moves")
        print(f"{self.agent1.__class__.__name__} Patterns: {self.agent1_patterns}")
        print(f"{self.agent2.__class__.__name__} Patterns: {self.agent2_patterns}")
    
    
    def _plot_results(self, num_games):
        '''
        the privat method [_print_results] can be used to plot the result of the messuremnts of one
        game simmulation as graphs.
        The parameter [num_games] is the amount of games the simulator simulated
        '''
        
        # create a plot figure in witch the graphs will take place
        plt.figure(figsize=(18, 12))
        
        # plot winrates
        plt.subplot(2, 3, 1)
        labels = [
            f"{self.agent1.__class__.__name__}",
            f"{self.agent2.__class__.__name__}",
            "Draws"
        ]
        plt.bar(labels, [self.agent1_wins, self.agent2_wins, self.draws], color=['blue', 'orange', 'green'])
        plt.title('Win Rates')
        plt.ylabel('Number of Games')

        # plot game lenghts
        plt.subplot(2, 3, 2)
        plt.hist(self.game_lengths, bins=range(min(self.game_lengths), max(self.game_lengths)+1), alpha=0.75, color='purple')
        plt.title('Game Length Distribution')
        plt.xlabel('Moves')
        plt.ylabel('Frequency')

        # plot winning patterns
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

        # plot memory usage as graph
        plt.subplot(2, 3, 4)
        plt.plot(range(1, num_games+1), self.memory_usages, marker='o', color='red')
        plt.title('Memory Usage')
        plt.xlabel('Game')
        plt.ylabel('MB')
        plt.grid(True)

        # plot execution time as graph
        plt.subplot(2, 3, 5)
        plt.plot(range(1, num_games+1), self.execution_times, marker='o', color='green')
        plt.title('Execution Time')
        plt.xlabel('Game')
        plt.ylabel('Seconds')
        plt.grid(True)

        plt.tight_layout()
        plt.show()