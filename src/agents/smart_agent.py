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
        if symbol == PLAYER_ONE_SYMBOL:
            self.opponent_symbol = PLAYER_TWO_SYMBOL
        else:
            self.opponent_symbol = PLAYER_ONE_SYMBOL

    def get_move(self, board):
        
        move = self._check_winning_move(board, self.symbol)
        if move is not None:
            return move
        move = self._check_winning_move(board, self.opponent_symbol)
        if move is not None:
            return move
        return self._random_move(board)

    def _check_winning_move(self, board, token):
        for col in range(self.cols):
            if self._is_valid_move(board, col):
                row = self._get_next_open_row(board, col)
                board.insert_token(col, token)
                if board.check_winner(token):
                    board.board[row][col] = " "
                    return col
                board.board[row][col] = " "
        return None

    def _random_move(self, board):
        possible_cols = []
        
        for col in range(self.cols):
            if board[0][col] == " ":
                possible_cols.append(col)
        
        random_number = rd.randint(0, len(possible_cols) - 1)
        random_move = possible_cols[random_number]
        
        return random_move
    
    def _is_valid_move(self, board,  col):
        return board.board[0][col] == " "

    def _get_next_open_row(self, board, col):
        for row in range(self.rows - 1, -1, -1):
            if board.board[row][col] == " ":
                return row
        return None