import random as rd

class RandomAgent:
    
    def __init__(self, rows, cols):
        # attributes [rows] & [cols] are used to get the deffined size of the board
        self.cols = cols
        self.rows = rows
        
    
    def get_possible_moves(self, board):
        '''
        the method [get_possible_moves] interates over the top elements of each colums [cols]
        and checks if there is a free space. if so it adds this colum to the list [possible_cols].
        The return value is the list [possible_cols].
        '''
    
        possible_cols = []
        
        for col in range(self.cols):
            # checks the top element of each col and append it to [possible_cols] if its empty.
            if board[0][col] == " ":
                possible_cols.append(col)
        
        return possible_cols
    
    
    def play_move():
        return False