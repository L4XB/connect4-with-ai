from constants import *
from agents.alpha_zero_agent.agent import AlphaZeroAgent
from agents.mini_max_agent import MiniMaxAgent

def evaluate_against_minimax(alpha_zero_agent, minimax_agent, num_games=100):
    wins = 0
    for _ in range(num_games):
        board = [[' ' for _ in range(7)] for _ in range(6)]
        player = PLAYER_ONE_SYMBOL
        while True:
            if player == alpha_zero_agent.symbol:
                col = alpha_zero_agent.get_move(board)
            else:
                col = minimax_agent.get_move(board)
            
            for row in reversed(range(6)):
                if board[row][col] == ' ':
                    board[row][col] = player
                    break
            
            winner = minimax_agent._check_winner(board)
            if winner == alpha_zero_agent.symbol:
                wins += 1
                break
            elif winner or all(cell != ' ' for row in board for cell in row):
                break
            
            player = PLAYER_TWO_SYMBOL if player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
    return wins / num_games

alpha_zero_agent = AlphaZeroAgent(rows= AMOUNT_ROWS, cols= AMOUNT_COLUMNS, symbol=PLAYER_ONE_SYMBOL, model_path="alpha_zero_model.pth")
mini_max_agent = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL, )
evaluate_against_minimax(alpha_zero_agent, mini_max_agent)