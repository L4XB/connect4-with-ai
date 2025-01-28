import numpy as np
from tqdm import tqdm
from agents.rl_agent.agent import RLAgent
from agents.mini_max_agent import MiniMaxAgent
from enviroment.game_board import GameBoard
from constants import *
import time
import matplotlib.pyplot as plt
import psutil
import os

def evaluate(num_games=100):
    # Initialisierung
    rl_agent = RLAgent(PLAYER_ONE_SYMBOL)
    rl_agent.load_model()
    minimax = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL, max_depth=4)
    
    # Tracking-Variablen
    results = {'RL_Wins': 0, 'MiniMax_Wins': 0, 'Draws': 0}
    game_lengths = []
    rl_winning_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    minimax_winning_patterns = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    execution_times = []
    memory_usages = []

    # Evaluationsloop
    for game in tqdm(range(num_games), desc="Evaluation"):
        start_time = time.time()
        env = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
        done = False
        turn = np.random.choice([0, 1])  # Zufälliger Startspieler
        
        while not done:
            if turn % 2 == 0:
                # RL Agent
                valid_moves = [c for c in range(AMOUNT_COLUMNS) if env.board[0][c] == ' ']
                action = rl_agent.get_action(env.board, valid_moves, training=False)
                env.insert_token(action, PLAYER_ONE_SYMBOL)
                if env.check_winner(PLAYER_ONE_SYMBOL):
                    results['RL_Wins'] += 1
                    winning_pattern = env.get_winning_pattern(PLAYER_ONE_SYMBOL)
                    rl_winning_patterns[winning_pattern] += 1
                    done = True
            else:
                # MiniMax Agent
                action = minimax.get_move(env.board)
                env.insert_token(action, PLAYER_TWO_SYMBOL)
                if env.check_winner(PLAYER_TWO_SYMBOL):
                    results['MiniMax_Wins'] += 1
                    winning_pattern = env.get_winning_pattern(PLAYER_TWO_SYMBOL)
                    minimax_winning_patterns[winning_pattern] += 1
                    done = True
            
            if env.is_draw():
                results['Draws'] += 1
                done = True
            
            turn += 1
        
        # Tracking
        game_lengths.append(turn)
        execution_times.append(time.time() - start_time)
        
        # Speichernutzung messen
        process = psutil.Process(os.getpid())
        memory_usages.append(process.memory_info().rss / (1024 * 1024))  # in MB

    # Ergebnisse anzeigen
    print("\nEvaluationsergebnisse:")
    print(f"RL Siege: {results['RL_Wins']} ({results['RL_Wins']/num_games*100:.1f}%)")
    print(f"MiniMax Siege: {results['MiniMax_Wins']} ({results['MiniMax_Wins']/num_games*100:.1f}%)")
    print(f"Unentschieden: {results['Draws']} ({results['Draws']/num_games*100:.1f}%)")
    print(f"Durchschnittliche Zuganzahl pro Spiel: {np.mean(game_lengths):.1f}")

    # Genauigkeitsmetriken
    rl_win_rate = results['RL_Wins'] / num_games
    minimax_win_rate = results['MiniMax_Wins'] / num_games
    draw_rate = results['Draws'] / num_games

    print("\nAccuracy Metrics:")
    print(f"RL Agent Win Rate: {rl_win_rate:.2f}")
    print(f"MiniMax Agent Win Rate: {minimax_win_rate:.2f}")
    print(f"Draw Rate: {draw_rate:.2f}")

    # Effizienzmetriken
    avg_execution_time = np.mean(execution_times)
    print(f"\nEfficiency Metrics:")
    print(f"Average Execution Time per Game: {avg_execution_time:.2f} seconds")

    # Spiel-Level-Metriken
    avg_game_length = np.mean(game_lengths)
    print(f"\nGame-Level Metrics:")
    print(f"Average Game Length: {avg_game_length:.2f} moves")
    print(f"RL Agent Winning Patterns: {rl_winning_patterns}")
    print(f"MiniMax Agent Winning Patterns: {minimax_winning_patterns}")

    # Ressourcennutzungsmetriken
    avg_memory_usage = np.mean(memory_usages)
    max_memory_usage = np.max(memory_usages)
    print(f"\nResource Utilization Metrics:")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")
    print(f"Maximum Memory Usage: {max_memory_usage:.2f} MB")

    # Grafiken erstellen
    plt.figure(figsize=(18, 12))

    # Plot 1: Gewinnraten
    plt.subplot(2, 3, 1)
    plt.bar(['RL Agent', 'MiniMax Agent', 'Draws'], [results['RL_Wins'], results['MiniMax_Wins'], results['Draws']], color=['blue', 'orange', 'green'])
    plt.title('Win Rates')
    plt.ylabel('Number of Games')

    # Plot 2: Spielzuganzahl-Verteilung
    plt.subplot(2, 3, 2)
    plt.hist(game_lengths, bins=range(min(game_lengths), max(game_lengths) + 1), alpha=0.75, color='purple')
    plt.title('Game Length Distribution')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')

    # Plot 3: Gewinnmuster
    plt.subplot(2, 3, 3)
    patterns = ['horizontal', 'vertical', 'diagonal']
    rl_pattern_counts = [rl_winning_patterns[p] for p in patterns]
    minimax_pattern_counts = [minimax_winning_patterns[p] for p in patterns]

    bar_width = 0.35
    index = range(len(patterns))

    plt.bar(index, rl_pattern_counts, bar_width, color='blue', label='RL Agent')
    plt.bar([i + bar_width for i in index], minimax_pattern_counts, bar_width, color='orange', label='MiniMax Agent')

    plt.title('Winning Patterns')
    plt.xlabel('Pattern Type')
    plt.ylabel('Frequency')
    plt.xticks([i + bar_width / 2 for i in index], patterns)
    plt.legend()

    # Plot 4: Speichernutzung über Spiele
    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_games + 1), memory_usages, marker='o', color='red')
    plt.title('Memory Usage Over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)

    # Plot 5: Ausführungszeit über Spiele
    plt.subplot(2, 3, 5)
    plt.plot(range(1, num_games + 1), execution_times, marker='o', color='green')
    plt.title('Execution Time Over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Beispielaufruf
evaluate(num_games=100)