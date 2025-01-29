from constants import *
from agents.alpha_zero_agent.agent import AlphaZeroAgent
from agents.mini_max_agent import MiniMaxAgent

def evaluate_against_minimax(alpha_zero_agent, minimax_agent, num_games=100):
    wins = 0
    for game in range(num_games):
        print(f"Evaluation game {game + 1}/{num_games}")
        board = [[' ' for _ in range(7)] for _ in range(6)]
        player = PLAYER_ONE_SYMBOL
        while True:
            if player == alpha_zero_agent.symbol:
                col = alpha_zero_agent.get_move(board)
            else:
                col = minimax_agent.get_move(board)
            
            # Spielzug ausführen
            for row in reversed(range(6)):
                if board[row][col] == ' ':
                    board[row][col] = player
                    break
            
            # Gewinner überprüfen
            winner = minimax_agent._check_winner(board)
            if winner == alpha_zero_agent.symbol:
                wins += 1
                break
            elif winner or all(cell != ' ' for row in board for cell in row):
                break
            
            # Spieler wechseln
            player = PLAYER_TWO_SYMBOL if player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
    
    winrate = wins / num_games
    return winrate