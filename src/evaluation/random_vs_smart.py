from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from enviroment.game_board import GameBoard
from agents.random_agent import RandomAgent
from agents.smart_agent import SmartAgent
import time
import matplotlib.pyplot as plt
import psutil
import os

def simulate_games_random_vs_smart(num_games):
    '''
    The method [simulate_games_random_vs_smart] simulates a Connect4 game between the random and 
    the smart agent. 
    The parameter [num_games] can be used to set the amount of the games the agents play against
    each other.
    '''
    
    smart_wins = 0
    random_wins = 0
    draws = 0
    game_lengths = []
    smart_winning_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    random_winning_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    execution_times = []
    memory_usages = [] 

    for game in range(num_games):
        start_time = time.time()
        board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
        agents = {
            PLAYER_ONE_SYMBOL: RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL),
            PLAYER_TWO_SYMBOL: SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)
        }
        
        current_player = PLAYER_ONE_SYMBOL 
        moves = 0
        
        while True:
            moves += 1
            # make a move
            move = agents[current_player].get_move(board.board)
            board.insert_token(move, current_player)
            
            # check if someone wins
            if board.check_winner(current_player):
                if current_player == PLAYER_TWO_SYMBOL:
                    smart_wins += 1
                    winning_pattern = board.get_winning_pattern(current_player)
                    smart_winning_patterns[winning_pattern] += 1
                else:
                    random_wins += 1
                    winning_pattern = board.get_winning_pattern(current_player)
                    random_winning_patterns[winning_pattern] += 1
                break
            
            # check if it's a draw
            if board.is_draw():
                draws += 1
                break
            
            # change player
            current_player = PLAYER_TWO_SYMBOL if current_player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

        game_lengths.append(moves)
        execution_times.append(time.time() - start_time)
        
        # messre memory use
        process = psutil.Process(os.getpid())
        memory_usages.append(process.memory_info().rss / (1024 * 1024)) 

    # print result
    print(f"Ergebnis nach {num_games} Spielen:")
    print(f"Smart Agent {PLAYER_TWO_SYMBOL} Siege: {smart_wins}")
    print(f"Random Agent {PLAYER_ONE_SYMBOL} Siege: {random_wins}")
    print(f"Unentschieden: {draws}")

    ##### performance evaluation and analysis
    # accuracy metrics
    smart_win_rate = smart_wins / num_games
    random_win_rate = random_wins / num_games
    draw_rate = draws / num_games

    print(f"\nAccuracy Metrics:")
    print(f"Smart Agent Win Rate: {smart_win_rate:.2f}")
    print(f"Random Agent Win Rate: {random_win_rate:.2f}")
    print(f"Draw Rate: {draw_rate:.2f}")

    # efficiency metrics
    avg_execution_time = sum(execution_times) / num_games
    print(f"\nEfficiency Metrics:")
    print(f"Average Execution Time per Game: {avg_execution_time:.2f} seconds")

    # game-level metrics
    avg_game_length = sum(game_lengths) / num_games
    print(f"\nGame-Level Metrics:")
    print(f"Average Game Length: {avg_game_length:.2f} moves")
    print(f"Smart Agent Winning Patterns: {smart_winning_patterns}")
    print(f"Random Agent Winning Patterns: {random_winning_patterns}")

    # resource utilization metrics
    avg_memory_usage = sum(memory_usages) / num_games
    max_memory_usage = max(memory_usages)
    print(f"\nResource Utilization Metrics:")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")
    print(f"Maximum Memory Usage: {max_memory_usage:.2f} MB")

    # plotting the results
    plt.figure(figsize=(18, 12))

    # plot 1: win rates
    plt.subplot(2, 3, 1)
    plt.bar(['Smart Agent', 'Random Agent', 'Draws'], [smart_wins, random_wins, draws], color=['blue', 'orange', 'green'])
    plt.title('Win Rates')
    plt.ylabel('Number of Games')

    # plot 2: game length distribution
    plt.subplot(2, 3, 2)
    plt.hist(game_lengths, bins=range(min(game_lengths), max(game_lengths) + 1), alpha=0.75, color='purple')
    plt.title('Game Length Distribution')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')

    # plot 3: winning patterns
    plt.subplot(2, 3, 3)
    patterns = ['horizontal', 'vertical', 'diagonal']
    smart_pattern_counts = [smart_winning_patterns[p] for p in patterns]
    random_pattern_counts = [random_winning_patterns[p] for p in patterns]

    bar_width = 0.35
    index = range(len(patterns))

    plt.bar(index, smart_pattern_counts, bar_width, color='blue', label='Smart Agent')
    plt.bar([i + bar_width for i in index], random_pattern_counts, bar_width, color='orange', label='Random Agent')

    plt.title('Winning Patterns')
    plt.xlabel('Pattern Type')
    plt.ylabel('Frequency')
    plt.xticks([i + bar_width / 2 for i in index], patterns)
    plt.legend()

    # plot 4: memory usage over games
    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_games + 1), memory_usages, marker='o', color='red')
    plt.title('Memory Usage Over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)

    # plot 5: execution time over games
    plt.subplot(2, 3, 5)
    plt.plot(range(1, num_games + 1), execution_times, marker='o', color='green')
    plt.title('Execution Time Over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()