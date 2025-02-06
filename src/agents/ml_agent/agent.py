import torch
from src.agents.ml_agent.model import Connect4CNN
from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
import copy

class AIAgent:
    def __init__(self, model_path, symbol):
        # initialize the neural network model
        self.model = Connect4CNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        # assign a symbol to the agent
        self.symbol = symbol
        
        # assigns the other symbol to the other player
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
        
        # set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # set the number of columns and rows
        self.cols = AMOUNT_COLUMNS
        self.rows = AMOUNT_ROWS

    def get_move(self, board):
        '''
        the method [get_move] uses the neural network model to decide what the best move is.
        It first checks for immediate winning or blocking moves, and if none are found,
        it uses the model to predict the best move.
        '''
        
        # check for winning move
        possible_moves = self._get_possible_moves(board)

        # check for winning move
        for move in range(len(possible_moves)):
            board_copy = copy.deepcopy(board)
            board_copy = self._play_move(board_copy, possible_moves[move], self.symbol)
            if self._is_winning_move(board_copy, self.symbol):
                return possible_moves[move]

        # check for move to block opponent
        for move in range(len(possible_moves)):
            board_copy = copy.deepcopy(board)
            board_copy = self._play_move(board_copy, possible_moves[move], self.opponent_symbol)
            if self._is_winning_move(board_copy, self.opponent_symbol):
                return possible_moves[move]

        # convert the board to a tensor and get the model's predictions
        state = self.board_to_tensor(board)
        with torch.no_grad():
            logits = self.model(state)

        # get the valid moves
        valid_moves = [col for col in range(7) if board[0][col] == " "]

        if not valid_moves:
            raise ValueError("No valid moves available.")

        # get the probabilities of the valid moves and select the best move
        move_probs = torch.softmax(logits[0][valid_moves], dim=0)
        selected_move = valid_moves[torch.argmax(move_probs).item()]

        return selected_move

    def board_to_tensor(self, board):
        '''
        the method [board_to_tensor] converts the board to a tensor that can be used as input to the neural network model.
        '''
        
        if not isinstance(board[0], list):
            board = [board[i:i+7] for i in range(0, len(board), 7)]

        channel_self = [[1.0 if cell == self.symbol else 0.0 for cell in row] for row in board]
        channel_opp = [[1.0 if cell == self.opponent_symbol else 0.0 for cell in row] for row in board]

        tensor = torch.FloatTensor([channel_self, channel_opp]).unsqueeze(0)
        return tensor.to(self.device)

    def _is_winning_move(self, board, tkn):
        """
        the method [_is_winning_move] checks if a player with a given token [tkn] has won,
        if so the method returns True, otherwise it returns False.
        A player wins if they have four of their tokens in a row, column, or diagonal.
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

    def _get_possible_moves(self, board):
        '''
        the private method [_get_possible_moves] iterates over the top elements of each column [cols]
        and checks if there is a free space. If so, it adds this column to the list [possible_cols].
        The return value is the list [possible_cols].
        '''

        possible_cols = []

        for col in range(self.cols):
            # checks the top element of each column and appends it to [possible_cols] if it's empty.
            if board[0][col] == " ":
                possible_cols.append(col)

        return possible_cols

    def _play_move(self, board, col, tkn):
        '''
        the method [_play_move] checks if it's possible to insert a token [tkn] in a column [col]
        and inserts the token in this column if it's possible.
        The method returns the board after the move is played.
        '''

        # checks if it's possible to set the token [tkn] in the specified column [col]
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