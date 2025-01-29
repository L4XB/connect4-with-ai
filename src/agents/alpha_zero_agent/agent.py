import torch
import numpy as np
import copy
from agents.alpha_zero_agent.model import ConnectFourNet
from constants import *
from agents.alpha_zero_agent.mcts import MCTS

class AlphaZeroAgent:
    def __init__(self, rows, cols, symbol, model_path=None, num_simulations=800, c_puct=1.5):
        self.rows = rows
        self.cols = cols
        self.symbol = symbol
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        self.model = ConnectFourNet(rows, cols)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.mcts = MCTS(self.model, rows, cols, num_simulations, c_puct)

    def get_move(self, board):
        col = self.mcts.run(board, self.symbol)
        return col