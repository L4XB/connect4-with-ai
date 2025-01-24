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
        # TODO: Implement
        return False

    def _is_winning_move(self, board, tkn):
        """
        the method [check_winner] checks if a player with a given token [tkn] has won, 
        if so the method return True if not the method return False.
        a player won if he has four of his tokens in a row/column or diagonal connected.
        """
        
        # checks horizontal lines
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(board[row][col + i] == tkn for i in range(4)):
                    return True

        # checks vertical lines
        for col in range(self.cols):
            for row in range(self.rows - 3):
                if all(board[row + i][col] == tkn for i in range(4)):
                    return True

        # check diagonals from the top right to the bottom left
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(board[row + i][col + i] == tkn for i in range(4)):
                    return True

        # checks diagonals from the left top to the right bottom
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(board[row + i][col - i] == tkn for i in range(4)):
                    return True

        # no winning line found
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
        # TODO: Implement
        return False

    def _get_next_open_row(self, board, col):
        # TODO: Implement
        return False