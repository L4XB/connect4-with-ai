import torch
import numpy as np
import copy
from agents.alpha_zero_agent.model import ConnectFourNet
from constants import *
from agents.alpha_zero_agent.mcts import MCTS

class AlphaZeroAgent:
    def __init__(self, rows, cols, symbol, model_path=None, num_simulations=100, c_puct=1.0):
        """
        Initialisiert den AlphaZero-Agenten.
        :param rows: Anzahl der Zeilen im Spielbrett.
        :param cols: Anzahl der Spalten im Spielbrett.
        :param symbol: Das Symbol des Agenten (z. B. 'X' oder 'O').
        :param model_path: Pfad zum trainierten Modell (falls vorhanden).
        :param num_simulations: Anzahl der MCTS-Simulationen pro Zug.
        :param c_puct: Exploration-Parameter für MCTS.
        """
        self.rows = rows
        self.cols = cols
        self.symbol = symbol
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # Lade das neuronale Netzwerk
        self.model = ConnectFourNet(rows, cols)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Initialisiere MCTS
        self.mcts = MCTS(self.model, rows, cols, num_simulations, c_puct)

    def get_move(self, board):
        """
        Berechnet den besten Zug für den aktuellen Spielzustand.
        :param board: Das aktuelle Spielbrett.
        :return: Die Spalte, in die der Agent setzt.
        """
        col = self.mcts.run(board, self.symbol)
        if col is None:
            raise ValueError("Kein gültiger Zug gefunden.")
        return col

    def _get_valid_moves(self, board):
        """
        Gibt die gültigen Züge für das aktuelle Brett zurück.
        :param board: Das aktuelle Spielbrett.
        :return: Ein Array mit gültigen Zügen (1 = gültig, 0 = ungültig).
        """
        return np.array([board[0][col] == ' ' for col in range(self.cols)], dtype=np.float32)

    def _play_move(self, board, col, symbol):
        """
        Führt einen Zug auf dem Brett aus.
        :param board: Das aktuelle Spielbrett.
        :param col: Die Spalte, in die gesetzt wird.
        :param symbol: Das Symbol des Spielers.
        :return: Das aktualisierte Brett.
        """
        new_board = copy.deepcopy(board)
        for row in reversed(range(self.rows)):
            if new_board[row][col] == ' ':
                new_board[row][col] = symbol
                return new_board
        return new_board

    def _is_terminal(self, board):
        """
        Überprüft, ob das Spiel beendet ist.
        :param board: Das aktuelle Spielbrett.
        :return: True, wenn das Spiel beendet ist, sonst False.
        """
        return self._check_winner(board) is not None or not self._get_valid_moves(board).any()

    def _check_winner(self, board):
        """
        Überprüft, ob ein Spieler gewonnen hat.
        :param board: Das aktuelle Spielbrett.
        :return: Das Symbol des Gewinners oder None, falls kein Gewinner.
        """
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if board[row][col] != ' ' and all(board[row][col+i] == board[row][col] for i in range(4)):
                    return board[row][col]
        for col in range(self.cols):
            for row in range(self.rows - 3):
                if board[row][col] != ' ' and all(board[row+i][col] == board[row][col] for i in range(4)):
                    return board[row][col]
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if board[row][col] != ' ' and all(board[row+i][col+i] == board[row][col] for i in range(4)):
                    return board[row][col]
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if board[row][col] != ' ' and all(board[row-i][col+i] == board[row][col] for i in range(4)):
                    return board[row][col]
        return None
