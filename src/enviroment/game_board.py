from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class GameBoard:
    def __init__(self, rows, cols):
        # attributes [rows] & [cols] are used to set the size of the board
        self.rows = rows
        self.cols = cols
        
        # generates a matrix with the size [rows] * [cols]
        self.board = [[" " for _ in range(cols)] for _ in range(rows)]

    def draw_board(self):
        '''
        the method [draw_board] draws the board with 
        the informations in [rows] & [cols]
        '''
        
        # creates the first top border of the board in relation to the amount in [cols]
        print("+" + ("---+" * self.cols))

        # builds the boards between the elements in the rows
        for row in self.board:
            row_str = "|"
            for cell in row:
                row_str += f" {cell} |"  
            print(row_str)
            
            # creates border in relation to the amount in [cols]
            print("+" + ("---+" * self.cols))


    def insert_token(self, col, tkn):
        '''
        the method [insert_token] checks
        if its possible to insert a token [tkn] in a Column [col] 
        and insert the token in this column if its possbile.
        The method return True if it was succesfull and False if not
        '''
        
        # checks if its possbile to set the token [tkn] in the specified column [col]
        if col < 0 or col >= self.cols:
            print(f"Invalid column: {col}")
            return False

        # loops over the board from bottom to top
        for row in range(self.rows - 1, -1, -1):
            # checks if the current element is empty
            if self.board[row][col] == " ":
                # sets the token at the empty place in the column
                self.board[row][col] = tkn
                # returns True so that the method execution is stopped
                return True

        print(f"Column {col} is full.")
        return False
    
    def check_winner(self, tkn):
        """
        the method [check_winner] checks if a player with a given token [tkn] has won, 
        if so the method return True if not the method return False.
        a player won if he has four of his tokens in a row/column or diagonal connected.
        """
        
        # checks horizontal lines
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row][col + i] == tkn for i in range(4)):
                    return True

        # checks vertical lines
        for col in range(self.cols):
            for row in range(self.rows - 3):
                if all(self.board[row + i][col] == tkn for i in range(4)):
                    return True

        # check diagonals from the top right to the bottom left
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i][col + i] == tkn for i in range(4)):
                    return True

        # checks diagonals from the left top to the right bottom
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(self.board[row + i][col - i] == tkn for i in range(4)):
                    return True

        # no winning line found
        return False

    def is_board_full(self):
        '''
        the method [is_board_full] if the playing board is full if so the method return True
        if the board is not full the method return False
        '''
        
        # loop over ever element and check if there is an empty one in the board
        for col in range(self.cols):
            for row in range(self.rows):
                if self.board[col][row] == " ":
                    return False
        
        # return True if there is no empty space in the boarx
        return True
        
    def is_draw(self):
        '''
        the method [is_draw] checks if the current game state is a draw and the game has ended.
        The method return True if no one of the player wo and the board is full otherwise the method
        returns False
        '''
        
        player_one_winner = self.check_winner(PLAYER_ONE_SYMBOL)
        player_two_winner = self.check_winner(PLAYER_TWO_SYMBOL)
        board_is_full = self.is_board_full()
        
        if not player_one_winner  and not player_two_winner and board_is_full:
            return True
        
        return False
        