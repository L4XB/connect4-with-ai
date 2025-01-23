class GameBoard:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [[' ' for _ in range(cols)] for _ in range(rows)]

    def draw_board(self):
        print('+' + ('---+' * self.cols))

        for row in self.board:
            row_str = '|'
            for cell in row:
                row_str += f' {cell} |'  
            print(row_str)

            print('+' + ('---+' * self.cols))

    def insert_token(self, col, token):
        if col < 0 or col >= self.cols:
            print(f"Ungültige Spalte: {col}")
            return False

        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == ' ':
                self.board[row][col] = token
                return True

        print(f"Spalte {col} ist voll.")
        return False
