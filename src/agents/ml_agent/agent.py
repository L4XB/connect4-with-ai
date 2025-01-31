import torch
from src.agents.ml_agent.model import Connect4CNN
from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class AIAgent:
    def __init__(self, model_path, symbol):
        self.model = Connect4CNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.symbol = symbol
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

    def get_move(self, board):
        state = self.board_to_tensor(board)
        with torch.no_grad():
            logits = self.model(state)
        
        valid_moves = [col for col in range(7) if board[0][col] == " "]
        move_probs = torch.softmax(logits[0][valid_moves], dim=0)
        return valid_moves[torch.argmax(move_probs).item()]

    def board_to_tensor(self, board):
        # 2D-Repräsentation mit zwei Kanälen
        channel_self = [[1.0 if cell == self.symbol else 0.0 for cell in row] for row in board]
        channel_opp = [[1.0 if cell == self.opponent_symbol else 0.0 for cell in row] for row in board]
        return torch.FloatTensor([channel_self, channel_opp]).unsqueeze(0)