from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from enviroment.game_board import GameBoard
from agents.smart_agent import SmartAgent
from agents.mini_max_agent import MiniMaxAgent 
import time
import matplotlib.pyplot as plt
import psutil 
import os

def simulate_games_minimax_vs_smart(num_games):
    '''
    The method [simulate_games_minimax_vs_smart] simulates a Connect4 game between the MiniMax and 
    the Smart Agent. 
    The parameter [num_games] can be used to set the amount of the games the agents play against
    each other.
    '''
    
    minimax_wins = 0
    smart_wins = 0
    draws = 0
    game_lengths = []
    minimax_winning_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    smart_winning_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    execution_times = []
    memory_usages = [] 

    for game in range(num_games):
        start_time = time.time()
        board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
        agents = {
            PLAYER_ONE_SYMBOL: MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL),
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
                if current_player == PLAYER_ONE_SYMBOL:
                    minimax_wins += 1
                    winning_pattern = board.get_winning_pattern(current_player)
                    minimax_winning_patterns[winning_pattern] += 1
                else:
                    smart_wins += 1
                    winning_pattern = board.get_winning_pattern(current_player)
                    smart_winning_patterns[winning_pattern] += 1
                break
            
            # check if it's a draw
            if board.is_draw():
                draws += 1
                break
            
            # change player
            current_player = PLAYER_TWO_SYMBOL if current_player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

        game_lengths.append(moves)
        execution_times.append(time.time() - start_time)
        
        # Speichernutzung messen
        process = psutil.Process(os.getpid())
        memory_usages.append(process.memory_info().rss / (1024 * 1024))  # Speichernutzung in MB

    # print result
    print(f"Ergebnis nach {num_games} Spielen:")
    print(f"Minimax Agent ({PLAYER_ONE_SYMBOL}) Siege: {minimax_wins}")
    print(f"Smart Agent ({PLAYER_TWO_SYMBOL}) Siege: {smart_wins}")
    print(f"Unentschieden: {draws}")

    ##### Performance Evaluation and Analysis
    # accuracy Metrics
    minimax_win_rate = minimax_wins / num_games
    smart_win_rate = smart_wins / num_games
    draw_rate = draws / num_games

    print(f"\nAccuracy Metrics:")
    print(f"Minimax Agent Win Rate: {minimax_win_rate:.2f}")
    print(f"Smart Agent Win Rate: {smart_win_rate:.2f}")
    print(f"Draw Rate: {draw_rate:.2f}")

    # efficiency metrics
    avg_execution_time = sum(execution_times) / num_games
    print(f"\nEfficiency Metrics:")
    print(f"Average Execution Time per Game: {avg_execution_time:.2f} seconds")

    # game-level metrics
    avg_game_length = sum(game_lengths) / num_games
    print(f"\nGame-Level Metrics:")
    print(f"Average Game Length: {avg_game_length:.2f} moves")
    print(f"Minimax Agent Winning Patterns: {minimax_winning_patterns}")
    print(f"Smart Agent Winning Patterns: {smart_winning_patterns}")

    # resource utilization metrics
    avg_memory_usage = sum(memory_usages) / num_games
    max_memory_usage = max(memory_usages)
    print(f"\nResource Utilization Metrics:")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")
    print(f"Maximum Memory Usage: {max_memory_usage:.2f} MB")

    # plotting the results
    plt.figure(figsize=(18, 12))

    # Plot 1: Win Rates
    plt.subplot(2, 3, 1)
    plt.bar(['Minimax Agent', 'Smart Agent', 'Draws'], [minimax_wins, smart_wins, draws], color=['blue', 'orange', 'green'])
    plt.title('Win Rates')
    plt.ylabel('Number of Games')

    # Plot 2: Game Length Distribution
    plt.subplot(2, 3, 2)
    plt.hist(game_lengths, bins=range(min(game_lengths), max(game_lengths) + 1), alpha=0.75, color='purple')
    plt.title('Game Length Distribution')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')

    # Plot 3: Winning Patterns
    plt.subplot(2, 3, 3)
    patterns = ['horizontal', 'vertical', 'diagonal']
    minimax_pattern_counts = [minimax_winning_patterns[p] for p in patterns]
    smart_pattern_counts = [smart_winning_patterns[p] for p in patterns]

    bar_width = 0.35
    index = range(len(patterns))

    plt.bar(index, minimax_pattern_counts, bar_width, color='blue', label='Minimax Agent')
    plt.bar([i + bar_width for i in index], smart_pattern_counts, bar_width, color='orange', label='Smart Agent')

    plt.title('Winning Patterns')
    plt.xlabel('Pattern Type')
    plt.ylabel('Frequency')
    plt.xticks([i + bar_width / 2 for i in index], patterns)
    plt.legend()

    # Plot 4: Memory Usage Over Games
    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_games + 1), memory_usages, marker='o', color='red')
    plt.title('Memory Usage Over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)

    # Plot 5: Execution Time Over Games
    plt.subplot(2, 3, 5)
    plt.plot(range(1, num_games + 1), execution_times, marker='o', color='green')
    plt.title('Execution Time Over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

