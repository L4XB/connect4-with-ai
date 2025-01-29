import unittest
from src.agents.mini_max_agent import MiniMaxAgent
from src.constants import AMOUNT_COLUMNS, AMOUNT_ROWS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class TestMiniMaxAgentMethods(unittest.TestCase):
    
    def __init__(self, methodName = "runMiniMaxAgentTests"):
        super().__init__(methodName)
        
        # create new MiniMaxAgent object to test it
        self.minimax_agent = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL, max_depth=3)
    
    def testInitialAgentSetup(self):
        '''
        the test method [testInitialAgentSetup] tests if the MiniMaxAgent gets correctly initialized,
        when a new MiniMaxAgent object is created
        '''
        
        self.assertEqual(self.minimax_agent.rows, AMOUNT_ROWS)
        self.assertEqual(self.minimax_agent.cols, AMOUNT_COLUMNS)
        self.assertEqual(self.minimax_agent.symbol, PLAYER_ONE_SYMBOL)
        self.assertEqual(self.minimax_agent.opponent_symbol, PLAYER_TWO_SYMBOL)
        self.assertEqual(self.minimax_agent.max_depth, 3)
    
    def testGetAllLinesMethod(self):
        '''
        the test method [testGetAllLinesMethod] checks if the method correctly identifies
        all possible lines (horizontal, vertical, diagonal) on the board
        '''
        
        test_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", "●", " ", " ", " ", " "],
            [" ", " ", "●", " ", " ", " ", " "],
            [" ", " ", "●", " ", " ", " ", " "],
            ["○", "○", "○", " ", " ", " ", " "]
        ]
        
        lines = self.minimax_agent._get_all_lines(test_board)
        
        # check if the number of lines is correct
        # horizontal lines: 6 rows * 4 possible positions = 24
        # vertical lines: 7 columns * 3 possible positions = 21
        # diagonal lines (top-left to bottom-right): 12 possible positions
        # diagonal lines (top-right to bottom-left): 12 possible positions
        # total: 24 + 21 + 12 + 12 = 69 possible lines
        self.assertEqual(len(lines), 69)
        
        # check if a known line is present (vertical line in column 2)
        vertical_line = [" ", "●", "●", "●"]
        self.assertTrue(any(line == vertical_line for line in lines))
        
        # check if a known line is present (horizontal line in bottom row)
        horizontal_line = ["○", "○", "○", " "]
        self.assertTrue(any(line == horizontal_line for line in lines))
        
        # verify specific counts for each type of line
        horizontal_count = 0
        vertical_count = 0
        diagonal_count = 0
        
        for line in lines:
            # Count specific known lines to ensure correct generation
            if line == vertical_line:
                vertical_count += 1
            elif line == horizontal_line:
                horizontal_count += 1
        
        # we should find exactly one of each of our test lines
        self.assertEqual(vertical_count, 1, "Should find exactly one vertical test line")
        self.assertEqual(horizontal_count, 1, "Should find exactly one horizontal test line")
    
    
    def testEvaluateLineMethod(self):
        '''
        the test method [testEvaluateLineMethod] checks if the method correctly evaluates
        different line configurations according to the heuristic
        '''
        
        # test three opponent symbols
        opponent_line = [self.minimax_agent.opponent_symbol] * 3 + [" "]
        self.assertEqual(self.minimax_agent._evaluate_line(opponent_line), -100)
        
        # test three player symbols
        player_line = [self.minimax_agent.symbol] * 3 + [" "]
        self.assertEqual(self.minimax_agent._evaluate_line(player_line), 50)
        
        # test two player symbols
        two_player_line = [self.minimax_agent.symbol] * 2 + [" ", " "]
        self.assertEqual(self.minimax_agent._evaluate_line(two_player_line), 10)
        
        # test one player symbol
        one_player_line = [self.minimax_agent.symbol] + [" "] * 3
        self.assertEqual(self.minimax_agent._evaluate_line(one_player_line), 1)
        
        # test mixed line
        mixed_line = [self.minimax_agent.symbol, self.minimax_agent.opponent_symbol, " ", " "]
        self.assertEqual(self.minimax_agent._evaluate_line(mixed_line), 0)
    
    def testIsTerminalMethod(self):
        '''
        the test method [testIsTerminalMethod] checks if the method correctly identifies
        terminal states (win, loss, or draw)
        '''
        
        # test winning position for player
        winning_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", "●", "●", "●", " ", " ", " "]
        ]
        self.assertTrue(self.minimax_agent._is_terminal(winning_board))
        
        # test winning position for opponent
        opponent_winning_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["○", " ", " ", " ", " ", " ", " "],
            ["○", " ", " ", " ", " ", " ", " "],
            ["○", " ", " ", " ", " ", " ", " "],
            ["○", " ", " ", " ", " ", " ", " "]
        ]
        self.assertTrue(self.minimax_agent._is_terminal(opponent_winning_board))
        
        # test non-terminal position
        non_terminal_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", "●", " ", " ", " ", " "],
            ["●", "●", "○", "○", " ", " ", " "]
        ]
        self.assertFalse(self.minimax_agent._is_terminal(non_terminal_board))
    
    def testGetMoveMethodImmediateWin(self):
        '''
        the test method [testGetMoveMethodImmediateWin] checks if the agent correctly
        identifies and makes an immediate winning move when available
        '''
        
        # test immediate winning move
        board_with_winning_move = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", "●", "●", " ", " ", " ", " "]
        ]
        
        move = self.minimax_agent.get_move(board_with_winning_move)
        self.assertEqual(move, 3)
    
    def testGetMoveMethodImmediateBlock(self):
        '''
        the test method [testGetMoveMethodImmediateBlock] checks if the agent correctly
        identifies and blocks an immediate winning move by the opponent
        '''
        
        # test immediate blocking move
        board_with_blocking_move = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["○", "○", "○", " ", " ", " ", " "]
        ]
        
        move = self.minimax_agent.get_move(board_with_blocking_move)
        self.assertEqual(move, 3)
    
    def testPlayMoveMethod(self):
        '''
        the test method [testPlayMoveMethod] checks if the method correctly places
        a token on the board in the specified column
        '''
        
        empty_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        
        expected_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        result_board = self.minimax_agent._play_move(empty_board, 0, PLAYER_ONE_SYMBOL)
        self.assertEqual(result_board, expected_board)

unittest.main()