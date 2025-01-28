from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL
import random as rd
import copy

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
        '''
        the method [get_move] uses the follwonign rules to check for a possible move:
        Step 1: Check for a winning move.
        Step 2: If yes play it.
        Step 3: If not, check for a move to block the opponent’s game.
        Step 4: If yes block it.
        Step 5: In none of the above move exists, just place the disc on any empty cell.
        
        And returns the move who the agent decided for
        '''
        possible_moves = self._get_possible_moves(board)
        
        # check for winning move
        for move in range(len(possible_moves)):
            board_copy = copy.deepcopy(board)
            board_copy = self._play_move(board_copy, possible_moves[move], self.symbol)
            if self._is_winning_move(board_copy, self.symbol):
                print("Winning move:", possible_moves[move])
                return possible_moves[move]
        
        # check for move to block opponent
        for move in range(len(possible_moves)):
            board_copy = copy.deepcopy(board)
            board_copy = self._play_move(board_copy, possible_moves[move], self.opponent_symbol)
            if self._is_winning_move(board_copy, self.opponent_symbol):
                print("Blocking move:", possible_moves[move])
                return possible_moves[move]
        
        # random move
        random_move = self._random_move(board)
        print("Random move:", random_move)
        return random_move


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
        '''
        the method [_random_move] uses the method [_get_possible_moves] so get all possible moves
        and select a random possible move and return it.
        '''
        
        possible_cols = self._get_possible_moves(board)
        
        random_number = rd.randint(0, len(possible_cols) - 1)
        random_move = possible_cols[random_number]
        
        return random_move
    
    def _get_possible_moves(self, board):
        '''
        the privat method [_get_possible_moves] interates over the top elements of each colums [cols]
        and checks if there is a free space. if so it adds this colum to the list [possible_cols].
        The return value is the list [possible_cols].
        '''
    
        possible_cols = []
        
        for col in range(self.cols):
            # checks the top element of each col and append it to [possible_cols] if its empty.
            if board[0][col] == " ":
                possible_cols.append(col)
        
        return possible_cols

    def _play_move(self, board, col, tkn):
        '''
        the method [insert_token] checks
        if its possible to insert a token [tkn] in a Column [col] 
        and insert the token in this column if its possbile.
        The method return the board after the move is played
        '''
        
        # checks if its possbile to set the token [tkn] in the specified column [col]
        if col < 0 or col >= self.cols:
            print(f"Invalid column: {col}")
            return False

        # loops over the board from bottom to top
        for row in range(self.rows - 1, -1, -1):
            # checks if the current element is empty
            if board[row][col] == " ":
                # sets the token at the empty place in the column
                board[row][col] = tkn
                # returns board so that the method execution is stopped
                return board

        print(f"Column {col} is full.")
        return board