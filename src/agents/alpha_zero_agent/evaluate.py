from agents.alpha_zero_agent.agent import AlphaZeroAgent
from agents.mini_max_agent import MiniMaxAgent
from constants import *

def evaluate():
    az_agent = AlphaZeroAgent(6, 7, PLAYER_ONE_SYMBOL, "az_model.pth")
    mm_agent = MiniMaxAgent(6, 7, PLAYER_TWO_SYMBOL, max_depth=1)
    
    wins = 0
    for game in range(100):
        board = [[' ']*7 for _ in range(6)]
        player = PLAYER_ONE_SYMBOL
        
        while True:
            if player == PLAYER_ONE_SYMBOL:
                col = az_agent.get_move(board)
            else:
                col = mm_agent.get_move(board)
            
            # Play move
            for row in reversed(range(6)):
                if board[row][col] == ' ':
                    board[row][col] = player
                    break
            
            # Check winner
            winner = az_agent.mcts.check_winner(board)
            if winner == PLAYER_ONE_SYMBOL:
                wins += 1
                break
            elif winner or all(cell != ' ' for row in board for cell in row):
                break
                
            player = PLAYER_TWO_SYMBOL if player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
    
    print(f"Win rate: {wins/100:.2%}")

evaluate()