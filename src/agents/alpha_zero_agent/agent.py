import torch
import numpy as np
import copy
from agents.alpha_zero_agent.model import ConnectFourNet
from agents.alpha_zero_agent.mcts import MCTS
from constants import *

class AlphaZeroAgent:
    def __init__(self, rows, cols, symbol, model_path=None, num_simulations=200, c_puct=1.5):
        self.rows = rows
        self.cols = cols
        self.symbol = symbol
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        self.model = ConnectFourNet(rows, cols)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.mcts = MCTS(self.model, rows, cols, num_simulations, c_puct)

    def get_move(self, board):
        return self.mcts.run(board, self.symbol)