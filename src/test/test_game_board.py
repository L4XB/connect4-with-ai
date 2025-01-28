import unittest
from src.enviroment.game_board import GameBoard
from src.constants import AMOUNT_COLUMNS, AMOUNT_ROWS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class TestGameBoardMethods(unittest.TestCase):
    
    def __init__(self, methodName = "runGameBoardTests"):
        super().__init__(methodName)
        
        # create new GameBoard object to test it
        self.game_board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
    
    def testInitialBoardSetup(self):
        '''
        the test method [testInitialBoardSetup] tests if the gameboard gets correctly initalize,
        when a new GameBoard object is created
        '''
        
        # default game board layout
        default_board_layout = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        
        # checks if the deault game board layout matches the generated gameboard layout
        self.assertEqual(default_board_layout, self.game_board.board)
    
    
    def testResetBoardMethod(self):
        '''
        the test method [testResetBoardMethod] checks if the reset method in the GameBoard class
        works correctly and resets the GameBoard correctly
        '''
        
        # simulate that some tokens are set into the GameBoard
        self.game_board.board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", "●"],
            [" ", " ", " ", " ", "●", " ", " "],
            [" ", " ", "●", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        # the layout the bard should have after the reset
        default_board_layout = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        
        # call reset method
        self.game_board.reset()
        
        # checks if reset works correctly
        self.assertEqual(self.game_board.board, default_board_layout)
    
    
    def testInsertTokenMethodWhenInsertionIsPossible(self):
        '''
        the test method [testInsertTokenMethodWhenInsertionIsPossible] checks if the insertion of a token
        into the gameboard works correctly.
        '''
        
        # insert token and check if the method returns True
        self.assertTrue(self.game_board.insert_token(0, PLAYER_ONE_SYMBOL))
        
        # simulate how the board should look like after the insertion
        default_board_layout_with_one_token = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        self.assertEqual(self.game_board.board, default_board_layout_with_one_token)
        self.game_board.reset()
        
    
    def testInsertTokenMethodWhenInsertionIsNotPossible(self):
        
        default_board_layout_with_a_full_line_of_tokens = [
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        self.game_board.board = default_board_layout_with_a_full_line_of_tokens
        
        self.assertFalse(self.game_board.insert_token(0, PLAYER_ONE_SYMBOL))
            
        self.game_board.reset()


    def testCheckWinnerMethod(self):
        return


unittest.main()