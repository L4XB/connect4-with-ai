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

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0

class MCTS:
    def __init__(self, model, rows, cols, num_simulations=200, c_puct=1.5):
        self.model = model
        self.rows = rows
        self.cols = cols
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None

    def run(self, board, player):
        self.root = Node(0)
        valid_moves = self.get_valid_moves(board)
        
        # Build initial tree
        state = self.board_to_state(board, player)
        with torch.no_grad():
            policy_logits, _ = self.model(state.unsqueeze(0))
        policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
        policy = policy * valid_moves
        policy /= policy.sum()
        
        for col in range(self.cols):
            if valid_moves[col]:
                self.root.children[col] = Node(policy[col])

        # Run simulations
        for _ in range(self.num_simulations):
            node = self.root
            current_board = copy.deepcopy(board)
            current_player = player
            path = []
            
            # Selection
            while node.children:
                action, node = self.select_child(node)
                path.append(node)
                current_board = self.play_move(current_board, action, current_player)
                current_player = self.get_opponent(current_player)
            
            # Expansion
            winner = self.check_winner(current_board)
            if not winner:
                valid = self.get_valid_moves(current_board)
                if valid.any():
                    state = self.board_to_state(current_board, current_player)
                    with torch.no_grad():
                        policy_logits, value = self.model(state.unsqueeze(0))
                    policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
                    policy = policy * valid
                    policy /= policy.sum()
                    
                    for col in range(self.cols):
                        if valid[col]:
                            node.children[col] = Node(policy[col], node)
            else:
                value = 1 if winner == player else -1
            
            # Backpropagation
            self.backpropagate(path, value)

        # Select best move
        visits = [self.root.children[col].visit_count if col in self.root.children else 0 
                 for col in range(self.cols)]
        return np.argmax(visits)

    def select_child(self, node):
        total_visits = sum(c.visit_count for c in node.children.values())
        best_score = -math.inf
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = child.value() + self.c_puct * child.prior * math.sqrt(total_visits + 1) / (child.visit_count + 1)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Alternate perspective

    def play_move(self, board, col, player):
        new_board = [row.copy() for row in board]
        for row in reversed(range(self.rows)):
            if new_board[row][col] == ' ':
                new_board[row][col] = player
                return new_board
        raise ValueError(f"Invalid move: {col}")

    def get_valid_moves(self, board):
        return np.array([board[0][col] == ' ' for col in range(self.cols)], dtype=np.float32)

    def check_winner(self, board):
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols-3):
                if board[row][col] != ' ' and board[row][col] == board[row][col+1] == board[row][col+2] == board[row][col+3]:
                    return board[row][col]
        # Vertical
        for col in range(self.cols):
            for row in range(self.rows-3):
                if board[row][col] != ' ' and board[row][col] == board[row+1][col] == board[row+2][col] == board[row+3][col]:
                    return board[row][col]
        # Diagonals
        for row in range(self.rows-3):
            for col in range(self.cols-3):
                if board[row][col] != ' ' and board[row][col] == board[row+1][col+1] == board[row+2][col+2] == board[row+3][col+3]:
                    return board[row][col]
        for row in range(3, self.rows):
            for col in range(self.cols-3):
                if board[row][col] != ' ' and board[row][col] == board[row-1][col+1] == board[row-2][col+2] == board[row-3][col+3]:
                    return board[row][col]
        return None

    def get_opponent(self, player):
        return PLAYER_TWO_SYMBOL if player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

    def board_to_state(self, board, player):
        state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        opponent = self.get_opponent(player)
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] == player:
                    state[0][r][c] = 1
                elif board[r][c] == opponent:
                    state[1][r][c] = 1
        return torch.tensor(state)