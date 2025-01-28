import math
import numpy as np
import copy
import torch
from constants import *

class Node:
    def __init__(self, prior, parent=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0

class MCTS:
    def __init__(self, model, rows, cols, num_simulations=100, c_puct=1.0):
        self.model = model
        self.rows = rows
        self.cols = cols
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None  # Initialize root as an attribute
    
    def run(self, board, player):
        self.root = Node(0)  # Store root as an attribute here
        state = self.board_to_state(board, player)
        
        with torch.no_grad():
            policy_logits, value = self.model(state.unsqueeze(0))
        policy = torch.exp(policy_logits).squeeze().numpy()
        valid_moves = self.get_valid_moves(board)
        policy *= valid_moves
        policy /= policy.sum() if policy.sum() else 1
        
        for col in range(self.cols):
            if valid_moves[col]:
                self.root.children[col] = Node(policy[col], self.root)
        
        for _ in range(self.num_simulations):
            node, current_board, current_player = self.root, copy.deepcopy(board), player
            search_path = [node]
            
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                current_board = self.play_move(current_board, action, current_player)
                current_player = self.get_opponent(current_player)
            
            if not self.is_terminal(current_board):
                state = self.board_to_state(current_board, current_player)
                with torch.no_grad():
                    policy_logits, value = self.model(state.unsqueeze(0))
                policy = torch.exp(policy_logits).squeeze().numpy()
                valid_moves = self.get_valid_moves(current_board)
                policy = policy * valid_moves
                policy /= policy.sum() if policy.sum() else 1
                
                for col in range(self.cols):
                    if valid_moves[col]:
                        node.children[col] = Node(policy[col], node)
            else:
                winner = self.check_winner(current_board)
                value = 1 if winner == player else -1 if winner else 0
            
            self.backpropagate(search_path, value, player)
        
        visit_counts = np.array([self.root.children[col].visit_count if col in self.root.children else 0 for col in range(self.cols)])
        return np.argmax(visit_counts)
    
    def select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score, best_action, best_child = -np.inf, None, None
        
        for action, child in node.children.items():
            score = child.value() + self.c_puct * child.prior * np.sqrt(total_visits + 1) / (child.visit_count + 1)
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        
        # Überprüfe, ob wir einen gültigen Zug haben
        if best_action is None:
            return None, None
        
        return best_action, best_child
    
    def backpropagate(self, path, value, player):
        for node in reversed(path):
            node.value_sum += value if node.parent and node.parent.parent else -value
            node.visit_count += 1
    
    def get_valid_moves(self, board):
        return np.array([board[0][col] == ' ' for col in range(self.cols)], dtype=np.float32)
    
    def play_move(self, board, action, current_player):
        """
        Spielt den Zug und gibt das neue Board zurück.
        :param board: Das aktuelle Board.
        :param action: Der ausgewählte Zug.
        :param current_player: Das Symbol des aktuellen Spielers.
        :return: Das neue Board nach dem Zug.
        """
        new_board = [row.copy() for row in board]  # Kopiere das Board, um es nicht zu verändern
        row, col = action  # Erwartet, dass action ein Tupel (row, col) ist
        
        # Überprüfe, ob row und col gültige Werte sind
        if row is None or col is None:
            raise ValueError(f"Ungültiger Zug: {action}. row und col dürfen nicht None sein.")
        
        if new_board[row][col] == ' ':
            new_board[row][col] = current_player
        else:
            raise ValueError(f"Ungültiger Zug: Feld {row}, {col} ist bereits besetzt.")
        
        return new_board

    
    def is_terminal(self, board):
        return self.check_winner(board) is not None or not self.get_valid_moves(board).any()
    
    def check_winner(self, board):
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
    
    def get_opponent(self, player):
        return PLAYER_TWO_SYMBOL if player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

    def board_to_state(self, board, player):
        """
        Wandelt das Spielbrett in ein Eingabeformat für das Modell um.
        Der Zustand besteht aus zwei Kanälen:
        - Kanal 1: Der aktuelle Spieler (1 für den eigenen Stein, -1 für den Stein des Gegners, 0 für leer).
        - Kanal 2: Der Gegner (1 für den Stein des Gegners, -1 für den eigenen Stein, 0 für leer).
        :param board: Das aktuelle Spielbrett.
        :param player: Das Symbol des aktuellen Spielers.
        :return: Ein Tensor mit der Form [2, 6, 7].
        """
        state = np.zeros((2, self.rows, self.cols))  # Zwei Kanäle für den Zustand

        for row in range(self.rows):
            for col in range(self.cols):
                if board[row][col] == player:
                    state[0, row, col] = 1  # Aktueller Spieler
                    state[1, row, col] = 0  # Gegner hat keine Steine
                elif board[row][col] == self.get_opponent(player):
                    state[0, row, col] = 0  # Aktueller Spieler hat keine Steine
                    state[1, row, col] = 1  # Gegner hat Steine
                else:
                    state[0, row, col] = 0  # Beide Kanäle sind 0 für leere Felder
                    state[1, row, col] = 0

        return torch.tensor(state, dtype=torch.float32)

