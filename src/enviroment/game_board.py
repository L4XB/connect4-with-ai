class GameBoard:
    def __init__(self, rows, cols):
        # attributes [rows] & [cols] are used to set the size of the board
        self.rows = rows
        self.cols = cols
        
        # generates a matrix with the size [rows] * [cols]
        self.board = [[' ' for _ in range(cols)] for _ in range(rows)]

    def draw_board(self):
        # the method [draw_board] draws the board with the informations in [rows] & [cols]
        
        # creates the first top border of the board in relation to the amount in [cols]
        print('+' + ('---+' * self.cols))

        # builds the boards between the elements in the rows
        for row in self.board:
            row_str = '|'
            for cell in row:
                row_str += f' {cell} |'  
            print(row_str)
            
            # creates border in relation to the amount in [cols]
            print('+' + ('---+' * self.cols))


    def insert_token(self, col, tkn):
        # the method [insert_token] checks
        # if its possible to insert a token [tkn] in a Column [col] 
        # and insert the token in this column if its possbile
        
        # checks if its possbile to set the token [tkn] in the specified column [col]
        if col < 0 or col >= self.cols:
            print(f"Invalid column: {col}")
            return False

        # loops over the board from bottom to top
        for row in range(self.rows - 1, -1, -1):
            # checks if the current element is empty
            if self.board[row][col] == ' ':
                # sets the token at the empty place in the column
                self.board[row][col] = tkn
                # returns True so that the method execution is stopped
                return True

        print(f"Column {col} is full.")
        return False
