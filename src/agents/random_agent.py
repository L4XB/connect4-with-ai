import random as rd
from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class RandomAgent:
    
    def __init__(self, rows, cols, symbol):
        # attributes [rows] & [cols] are used to get the deffined size of the board
        self.cols = cols
        self.rows = rows
        
        # assign a symbol to the agent
        self.symbol = symbol
        
        # assigns the other symbol to the other player
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
        
        
    
    def get_move(self, board):
        '''
        the method [get_move] reads over the private method [__get_possible_moves] all possible moves,
        picks a random move and returns this move.
        '''
        
        possible_moves = self._get_possible_moves(board)
        # generates a random number in the range 0 <= [random_number] lenght of possible moves -1
        random_number = rd.randint(0, len(possible_moves) - 1)
        random_move = possible_moves[random_number]
        
        return random_move
    
    
    def _get_possible_moves(self, board):
        '''
        the privat method [__get_possible_moves] interates over the top elements of each colums [cols]
        and checks if there is a free space. if so it adds this colum to the list [possible_cols].
        The return value is the list [possible_cols].
        '''
    
        possible_cols = []
        
        for col in range(self.cols):
            # checks the top element of each col and append it to [possible_cols] if its empty.
            if board[0][col] == " ":
                possible_cols.append(col)
        
        return possible_cols
    
    
