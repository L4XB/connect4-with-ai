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
    def __init__(self, model, rows, cols, num_simulations=800, c_puct=1.5):
        self.model = model
        self.rows = rows
        self.cols = cols
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None

    def run(self, board, player):
        self.root = Node(0)
        state = self.board_to_state(board, player)
        
        with torch.no_grad():
            policy_logits, _ = self.model(state.unsqueeze(0))
        policy = torch.exp(policy_logits).squeeze().numpy()
        valid_moves = self.get_valid_moves(board)
        policy = policy * valid_moves
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        
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
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy /= policy_sum
                
                for col in range(self.cols):
                    if valid_moves[col]:
                        node.children[col] = Node(policy[col], node)
                value = value.item()
            else:
                winner = self.check_winner(current_board)
                if winner == current_player:
                    value = 1
                elif winner is None:
                    value = 0
                else:
                    value = -1
            
            self.backpropagate(search_path, value)
        
        visit_counts = np.array([self.root.children[col].visit_count if col in self.root.children else 0 for col in range(self.cols)])
        return np.argmax(visit_counts)

    def select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -math.inf
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = child.value() + self.c_puct * child.prior * math.sqrt(total_visits) / (child.visit_count + 1)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def backpropagate(self, path, value):
        current_value = value
        for node in reversed(path):
            node.value_sum += current_value
            node.visit_count += 1
            current_value = -current_value

    def get_valid_moves(self, board):
        return np.array([board[0][col] == ' ' for col in range(self.cols)], dtype=np.float32)

    def play_move(self, board, col, symbol):
        new_board = [row.copy() for row in board]
        for row in reversed(range(self.rows)):
            if new_board[row][col] == ' ':
                new_board[row][col] = symbol
                return new_board
        return new_board

    def is_terminal(self, board):
        return self.check_winner(board) is not None or not self.get_valid_moves(board).any()

    def check_winner(self, board):
        # Implementierung der Gewinnüberprüfung
        pass

    def get_opponent(self, player):
        return PLAYER_TWO_SYMBOL if player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

    def board_to_state(self, board, player):
        state = np.zeros((2, self.rows, self.cols))
        opponent = self.get_opponent(player)
        for row in range(self.rows):
            for col in range(self.cols):
                if board[row][col] == player:
                    state[0][row][col] = 1
                elif board[row][col] == opponent:
                    state[1][row][col] = 1
        return torch.FloatTensor(state)