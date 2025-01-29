import unittest
from src.agents.smart_agent import SmartAgent
from src.constants import AMOUNT_COLUMNS, AMOUNT_ROWS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class TestSmartAgentMethods(unittest.TestCase):
    
    def __init__(self, methodName = "runSmartAgentTests"):
        super().__init__(methodName)
        
        # create new SmartAgent object to test it
        self.smart_agent = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
    
    def testInitialAgentSetup(self):
        '''
        the test method [testInitialAgentSetup] tests if the SmartAgent gets correctly initialized,
        when a new SmartAgent object is created
        '''
        
        self.assertEqual(self.smart_agent.rows, AMOUNT_ROWS)
        self.assertEqual(self.smart_agent.cols, AMOUNT_COLUMNS)
        self.assertEqual(self.smart_agent.symbol, PLAYER_ONE_SYMBOL)
        self.assertEqual(self.smart_agent.opponent_symbol, PLAYER_TWO_SYMBOL)
    
    def testGetPossibleMovesMethod(self):
        '''
        the test method [testGetPossibleMovesMethod] checks if the method correctly identifies
        possible moves on different board states
        '''
        
        # test on empty board - all columns should be possible
        empty_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        
        possible_moves = self.smart_agent._get_possible_moves(empty_board)
        self.assertEqual(possible_moves, [0, 1, 2, 3, 4, 5, 6])
        
        # test on board with some full columns
        partially_full_board = [
            ["●", " ", " ", " ", " ", " ", "○"],
            ["●", " ", " ", " ", " ", " ", "○"],
            ["●", " ", " ", " ", " ", " ", "○"],
            ["●", " ", " ", " ", " ", " ", "○"],
            ["●", " ", " ", " ", " ", " ", "○"],
            ["●", " ", " ", " ", " ", " ", "○"]
        ]
        
        possible_moves = self.smart_agent._get_possible_moves(partially_full_board)
        self.assertEqual(possible_moves, [1, 2, 3, 4, 5])
    
    def testPlayMoveMethod(self):
        '''
        the test method [testPlayMoveMethod] checks if the method correctly places
        a token on the board in the specified column
        '''
        
        # test placing token in empty column
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
        
        result_board = self.smart_agent._play_move(empty_board, 0, PLAYER_ONE_SYMBOL)
        self.assertEqual(result_board, expected_board)
        
        # test placing token on partially filled column
        partially_filled_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        expected_board_after_move = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        result_board = self.smart_agent._play_move(partially_filled_board, 0, PLAYER_ONE_SYMBOL)
        self.assertEqual(result_board, expected_board_after_move)
    
    def testIsWinningMoveMethod(self):
        '''
        the test method [testIsWinningMoveMethod] checks if the method correctly identifies
        winning positions in all possible directions (horizontal, vertical, diagonal)
        '''
        
        # test horizontal win
        horizontal_win_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", "●", "●", "●", " ", " ", " "]
        ]
        
        self.assertTrue(self.smart_agent._is_winning_move(horizontal_win_board, PLAYER_ONE_SYMBOL))
        
        # test vertical win
        vertical_win_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        self.assertTrue(self.smart_agent._is_winning_move(vertical_win_board, PLAYER_ONE_SYMBOL))
        
        # test diagonal win
        diagonal_win_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            [" ", " ", "●", " ", " ", " ", " "],
            [" ", "●", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        self.assertTrue(self.smart_agent._is_winning_move(diagonal_win_board, PLAYER_ONE_SYMBOL))
    
    def testGetMoveMethodWinningMove(self):
        '''
        the test method [testGetMoveMethodWinningMove] checks if the agent correctly
        identifies and makes winning moves when available
        '''
        
        # test horizontal winning opportunity
        board_with_winning_move = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", "●", "●", " ", " ", " ", " "]
        ]
        
        move = self.smart_agent.get_move(board_with_winning_move)
        self.assertEqual(move, 3)
    
    def testGetMoveMethodBlockingMove(self):
        '''
        the test method [testGetMoveMethodBlockingMove] checks if the agent correctly
        identifies and blocks opponent winning moves when available
        '''
        
        # test blocking opportunity
        board_with_blocking_move = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["○", "○", "○", " ", " ", " ", " "]
        ]
        
        move = self.smart_agent.get_move(board_with_blocking_move)
        self.assertEqual(move, 3)

unittest.main()