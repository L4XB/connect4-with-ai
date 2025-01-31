import random as rd
import copy
import math
from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class MiniMaxAgent:
    def __init__(self, rows, cols, symbol, max_depth = 2):
        # attributes [rows] & [cols] are used to get the deffined size of the board
        self.rows = rows
        self.cols = cols
        
        # assign a symbol to the agent
        self.symbol = symbol
        
        # assigns the other symbol to the other player
        self.opponent_symbol = PLAYER_TWO_SYMBOL if symbol == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL
        
        # number of moves that calculated bevor desicion
        self.max_depth = max_depth

    def get_move(self, board):
        '''
        the method [get_move] uses the Mini-Max-Algorithem to decide what the the move with the most
        value for the agent is.
        After checken for the best move, the method returns this move.
        '''
        
        # check if there is a way to immediatly win
        for col in self._get_possible_moves(board):
            if self._is_winning_move(board, col, self.symbol):
                return col
        
        # check if there is a way the block the opponent from winning
        for col in self._get_possible_moves(board):
            if self._is_winning_move(board, col, self.opponent_symbol):
                return col
        
        # set a variable to keep track of the best possible moves
        best_score = -math.inf
        best_moves = []
        
        # start with Mini-Max-Algorithem
        # interrate over all possible moves 
        for col in self._get_possible_moves(board):
            # create a copy of the current board
            board_copy = copy.deepcopy(board)
            
            # plays the first move in the possible moves 
            self._play_move(board_copy, col, self.symbol)
            
            # uses the minimax function to get a score/evaluation of the value of the current move
            score = self._minimax(board_copy, self.max_depth - 1, -math.inf, math.inf, False)
            
            # check if the score is higher that the current highest score. If so set it to the new best score
            if score > best_score:
                best_score = score
                best_moves = [col]
            elif score == best_score:
                best_moves.append(col)
        # return on of the best moves random, if there are no good moves it return a random move.
        return rd.choice(best_moves) if best_moves else self._random_move(board)

    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        '''
        the private method [_minimax] implements the minimax alorithm. The method checks all possible
        moves on a board [board] with the amount of steps [depth] in the future and evaluats them, to 
        pick the best possible move.
        [board] is the current gameboard
        [depth] is the maximum depth to wich the algorithmen searches
        [alpha] is the best current value the maximizer can guarantee
        [beta] is the best current value the minimizer can guarantee
        [maximizing_player] a Boolean value that indicates whether the current player 
        is the maximiser (True) or the minimiser (False).
        '''
        
        # if the game has ended or the search is finished the score of the
        # heuristic evaluation of the current board will be retuned
        if depth == 0 or self._is_terminal(board):
            return self._heuristic_evaluation(board)
            
        possible_moves = self._get_possible_moves(board)
        
        
        if maximizing_player:
            max_eval = -math.inf
            
            # interrates over all possible moves
            for col in possible_moves:
                
                # create a copy of the current game boarx
                board_copy = copy.deepcopy(board)
                
                # play the current move 
                self._play_move(board_copy, col, self.symbol)
                
                # calls the function [_minimax] recursively with the new game state
                eval = self._minimax(board_copy, depth - 1, alpha, beta, False)
                
                # updates the current maximal evaluation if the new evaluations is higher
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            
            # interrates over all possible moves
            for col in possible_moves:
                
                # create a copy of the current game board
                board_copy = copy.deepcopy(board)
                
                # play the current move 
                self._play_move(board_copy, col, self.opponent_symbol)
                
                 # calls the function [_minimax] recursively with the new game state
                eval = self._minimax(board_copy, depth - 1, alpha, beta, True)
                
                # updates the current minimal evaluation if the new evaluations is lower
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval


    def _heuristic_evaluation(self, board):
        '''
        The private method [_heuristic_evaluation] calculates a score of the evaluation of all lines
        and returns this score. It now includes:
        1. Line evaluations (horizontal, vertical, diagonal)
        2. Center control bonus
        3. Threat detection (immediate wins/blocks)
        '''
        
        score = 0
        
        # Evaluate all possible lines
        for line in self._get_all_lines(board):
            score += self._evaluate_line(line)
        
        # Add center control bonus
        center_columns = [2, 3, 4] 
        for row in range(self.rows):
            for col in center_columns:
                if board[row][col] == self.symbol:
                    score += 3 
                elif board[row][col] == self.opponent_symbol:
                    score -= 3 
        
        for col in range(self.cols):
            if self._is_winning_move(board, col, self.symbol):
                score += 1000
            if self._is_winning_move(board, col, self.opponent_symbol):
                score -= 1000
        
        return score


    def _evaluate_line(self, line):
        '''
        the private method [_evaluate_line] returns a evaluation of a line [line].
        It checks how many symbols the player have in a row in relation to the opponent.
        '''
        
        # checks how many symbols the player and the opponent have in a line
        player_count = line.count(self.symbol)
        opponent_count = line.count(self.opponent_symbol)
        
        # returns diffrent evaluations fr diffrent states.
        if opponent_count == 3 and player_count == 0: return -500
        if player_count == 3 and opponent_count == 0: return 100
        if player_count == 2 and opponent_count == 0: return 20
        if player_count == 1 and opponent_count == 0: return 2
        return 0


    def _get_all_lines(self, board):
        '''
        the private method [_get_all_lines] creates an list of all possible horizontal, 
        vertical and diagonal lines with the range of four elemets in a board [board]
        and returns this list afterwards.
        ''' 
        
        lines = []
        
        # adding horizontal lines
        for row in range(self.rows):
            for col in range(self.cols - 3):
                lines.append([board[row][col+i] for i in range(4)])
        
        # adding vertical lines
        for col in range(self.cols):
            for row in range(self.rows - 3):
                lines.append([board[row+i][col] for i in range(4)])
        
        # adding diagonals
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                lines.append([board[row+i][col+i] for i in range(4)])
                lines.append([board[row+3-i][col+i] for i in range(4)])
        
        # returns the list with all lines
        return lines

    def _is_terminal(self, board):
        '''
        the private method [_is_terminal] can be used to check if a game on a board [board] has
        endet by winning of one player or a draw
        '''
        return (self._check_winner(board, self.symbol) or 
                self._check_winner(board, self.opponent_symbol) or 
                len(self._get_possible_moves(board)) == 0)

    def _check_winner(self, board, symbol):
        '''
        the private method [_check_winner] checks if a player with the symbol [symbol]
        won on a board [board] and returns True if so. Otherwise it returns False
        '''
        
        # check horizontal lines
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(board[row][col+i] == symbol for i in range(4)):
                    return True
        # check vertical lines
        for col in range(self.cols):
            for row in range(self.rows - 3):
                if all(board[row+i][col] == symbol for i in range(4)):
                    return True
        # check diagonals
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
        '''
        the private method [_is_winning_move] checks if a move in a column [col] with the
        symbol [symbol] on a board [board] will lead to a win for the player with the
        given symbol
        '''
        
        temp_board = copy.deepcopy(board)
        if self._play_move(temp_board, col, symbol):
            return self._check_winner(temp_board, symbol)
        return False

    def _random_move(self, board):
        '''
        the private method [_random_move] returns a random possible move on 
        a given board [board]
        '''
        
        possible_cols = self._get_possible_moves(board)
        # retuns a random possible move if availible 
        return rd.choice(possible_cols) if possible_cols else None

    def _get_possible_moves(self, board):
        '''
        the private method [_get_possible_moves] returns all possible columns of a 
        board [board] in witch a symbol can be entered
        '''
        possible_moves = []
        for col in range(self.cols):
            if board[0][col] == ' ':
                possible_moves.append(col)
        return possible_moves

    def _play_move(self, board, col, symbol):
        '''
        the private methode [_play_move] sets a token [symbol] to a column [col] in a game board [board].
        '''
        
        # loops over the board from bottom to top
        for row in range(self.rows-1, -1, -1):
              # checks if the current element is empty
            if board[row][col] == " ":
                # sets the token at the empty place in the column
                board[row][col] = symbol
                # returns board so that the method execution is stopped
                return board
        return False