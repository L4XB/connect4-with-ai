import torch
from src.agents.ml_agent.model import Connect4CNN
from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class AIAgent:
    def __init__(self, model_path, symbol):
        self.model = Connect4CNN()
        self.model.load_state_dict(torch.load(model_path,weights_only = True, map_location=torch.device('cpu')))
        self.model.eval()
        self.symbol = symbol
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_move(self, board):
        state = self.board_to_tensor(board)
        with torch.no_grad():
            logits = self.model(state)

        valid_moves = [col for col in range(7) if board[0][col] == " "]

        if not valid_moves:
            raise ValueError("No valid moves available.")

        move_probs = torch.softmax(logits[0][valid_moves], dim=0)

        selected_move = valid_moves[torch.argmax(move_probs).item()]

        return selected_move

    def board_to_tensor(self, board):
        if not isinstance(board[0], list):
            board = [board[i:i+7] for i in range(0, len(board), 7)]

        channel_self = [[1.0 if cell == self.symbol else 0.0 for cell in row] for row in board]
        channel_opp = [[1.0 if cell == self.opponent_symbol else 0.0 for cell in row] for row in board]

        tensor = torch.FloatTensor([channel_self, channel_opp]).unsqueeze(0)
        return tensor.to(self.device)