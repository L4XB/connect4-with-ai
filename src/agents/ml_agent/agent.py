import torch
from src.agents.ml_agent.model import Connect4MLP

class AIAgent:
    def __init__(self, model_path, symbol):
        self.model = Connect4MLP()
        self.model.load_state_dict(torch.load(model_path, weights_only = True))
        self.model.eval()
        self.symbol = symbol

    def get_move(self, board):
        state = self.board_to_tensor(board)
        with torch.no_grad():
            logits = self.model(state)
        valid_moves = [col for col in range(7) if board[0][col] == " "]
        return valid_moves[logits[0][valid_moves].argmax().item()]

    def board_to_tensor(self, board):
        vec = []
        for row in board:
            for cell in row:
                if cell == self.symbol: vec.append(1)
                elif cell == " ": vec.append(0)
                else: vec.append(-1)
        return torch.FloatTensor(vec).unsqueeze(0)