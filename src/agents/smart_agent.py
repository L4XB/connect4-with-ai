from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL
import random as rd

class SmartAgent:

    def __init__(self, rows, cols, symbol):
        # attributes [rows] & [cols] are used to get the deffined size of the board
        self.rows = rows
        self.cols = cols
        
        # assign a symbol to the agent
        self.symbol = symbol
        
        # assigns the other symbol to the other player
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

    def get_move(self, board):
        return False

    def _check_winning_move(self, board, token):
        return False

    def _random_move(self, board):
        possible_cols = []
        
        for col in range(self.cols):
            if board[0][col] == " ":
                possible_cols.append(col)
        
        random_number = rd.randint(0, len(possible_cols) - 1)
        random_move = possible_cols[random_number]
        
        return random_move
    
    def _is_valid_move(self, board,  col):
        return False

    def _get_next_open_row(self, board, col):
        return False