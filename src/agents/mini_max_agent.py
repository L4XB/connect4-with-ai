import random as rd
import copy
import math
from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class MiniMaxAgent:
    def __init__(self, rows, cols, symbol, max_depth=4):
        self.rows = rows
        self.cols = cols
        self.symbol = symbol
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
        self.max_depth = max_depth

    def get_move(self, board):
        # Immediate win check
        for col in self._get_possible_moves(board):
            if self._is_winning_move(board, col, self.symbol):
                return col
        
        # Block opponent win
        for col in self._get_possible_moves(board):
            if self._is_winning_move(board, col, self.opponent_symbol):
                return col
        
        # Start Minimax search
        best_score = -math.inf
        best_moves = []
        
        for col in self._get_possible_moves(board):
            board_copy = copy.deepcopy(board)
            self._play_move(board_copy, col, self.symbol)
            score = self._minimax(board_copy, self.max_depth - 1, -math.inf, math.inf, False)
            
            if score > best_score:
                best_score = score
                best_moves = [col]
            elif score == best_score:
                best_moves.append(col)
        
        return rd.choice(best_moves) if best_moves else self._random_move(board)

    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self._is_terminal(board):
            return self._heuristic_evaluation(board)
            
        possible_moves = self._get_possible_moves(board)
        
        if maximizing_player:
            max_eval = -math.inf
            for col in possible_moves:
                board_copy = copy.deepcopy(board)
                self._play_move(board_copy, col, self.symbol)
                eval = self._minimax(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for col in possible_moves:
                board_copy = copy.deepcopy(board)
                self._play_move(board_copy, col, self.opponent_symbol)
                eval = self._minimax(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def _heuristic_evaluation(self, board):
        score = 0
        
        # Evaluate all possible lines
        for line in self._get_all_lines(board):
            score += self._evaluate_line(line)
        
        return score

    def _evaluate_line(self, line):
        player_count = line.count(self.symbol)
        opponent_count = line.count(self.opponent_symbol)
        
        if opponent_count == 3 and player_count == 0:
            return -100
        if player_count == 3 and opponent_count == 0:
            return 50
        if player_count == 2 and opponent_count == 0:
            return 10
        if player_count == 1 and opponent_count == 0:
            return 1
        return 0

    def _get_all_lines(self, board):
        lines = []
        
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                lines.append([board[row][col+i] for i in range(4)])
        
        # Vertical
        for col in range(self.cols):
            for row in range(self.rows - 3):
                lines.append([board[row+i][col] for i in range(4)])
        
        # Diagonals
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                lines.append([board[row+i][col+i] for i in range(4)])
                lines.append([board[row+3-i][col+i] for i in range(4)])
        
        return lines

    def _is_terminal(self, board):
        return (self._check_winner(board, self.symbol) or 
                self._check_winner(board, self.opponent_symbol) or 
                len(self._get_possible_moves(board)) == 0)

    def _check_winner(self, board, symbol):
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(board[row][col+i] == symbol for i in range(4)):
                    return True
        # Vertical
        for col in range(self.cols):
            for row in range(self.rows - 3):
                if all(board[row+i][col] == symbol for i in range(4)):
                    return True
        # Diagonals
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(board[row+i][col+i] == symbol for i in range(4)):
                    return True
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(board[row-i][col+i] == symbol for i in range(4)):
                    return True
        return False

    def _is_winning_move(self, board, col, symbol):
        temp_board = copy.deepcopy(board)
        if self._play_move(temp_board, col, symbol):
            return self._check_winner(temp_board, symbol)
        return False

    def _random_move(self, board):
        possible_cols = self._get_possible_moves(board)
        return rd.choice(possible_cols) if possible_cols else None

    def _get_possible_moves(self, board):
        return [col for col in range(self.cols) if board[0][col] == " "]

    def _play_move(self, board, col, symbol):
        for row in range(self.rows-1, -1, -1):
            if board[row][col] == " ":
                board[row][col] = symbol
                return True
        return False