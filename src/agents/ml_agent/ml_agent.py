import torch
import copy
from model import Connect4Model
from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class Connect4Agent:
    def __init__(self, model_path, symbol):
        self.model = self._load_model(model_path)
        self.symbol = symbol
        self.rows = 6
        self.cols = 7

    def _load_model(self, model_path):
        model = Connect4Model()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def get_best_move(self, game_board):
        valid_columns = self._get_valid_columns(game_board)
        best_score = -float('inf')
        best_column = None

        for col in valid_columns:
            simulated_board = self._simulate_move(game_board, col)
            if simulated_board is None:
                continue

            input_tensor = self._board_to_tensor(simulated_board)
            with torch.no_grad():
                output = self.model(input_tensor)
                win_prob = torch.softmax(output, dim=0)[0].item()

            if win_prob > best_score:
                best_score = win_prob
                best_column = col

        return best_column

    def _get_valid_columns(self, game_board):
        return [col for col in range(self.cols) if game_board.board[0][col] == " "]

    def _simulate_move(self, game_board, column):
        copied_board = copy.deepcopy(game_board)
        success = copied_board.insert_token(column, self.symbol)
        return copied_board if success else None

    def _board_to_tensor(self, game_board):
        flat_board = []
        for col in range(self.cols):
            for row in range(self.rows):
                cell = game_board.board[row][col]
                if cell == " ":
                    flat_board.append(0)
                elif cell == PLAYER_ONE_SYMBOL:
                    flat_board.append(1)
                else:
                    flat_board.append(-1)
        return torch.FloatTensor(flat_board)