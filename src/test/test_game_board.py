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
        
        # checks if the token is insered at the right row and the right place
        self.assertEqual(self.game_board.board, default_board_layout_with_one_token)
        
        # reset the gameboard
        self.game_board.reset()
        
    
    def testInsertTokenMethodWhenInsertionIsNotPossible(self):
        '''
        the test method [testInsertTokenMethodWhenInsertionIsNotPossible] checks if the insertion of a token
        into the gameboard throws an error if the column is full.
        '''
        
        # simulate full column
        default_board_layout_with_a_full_line_of_tokens = [
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        # set gameboard to simulated board
        self.game_board.board = default_board_layout_with_a_full_line_of_tokens
        
        # test if the method returns False because the column is full
        self.assertFalse(self.game_board.insert_token(0, PLAYER_ONE_SYMBOL))
        
        # reset the board
        self.game_board.reset()


    def testCheckWinnerMethod(self):
        '''
        the test method [testCheckWinnerMethod] checks if the board recognizes if a player with an
        specific token won.
        '''
        
        ## case one: check if vertical wins get recognized
        self.game_board.board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        # checks if a vertical win gets recognizes
        self.assertTrue(self.game_board.check_winner(PLAYER_ONE_SYMBOL))
        
        # reset board
        self.game_board.reset()
        
        
        ## case two: check if vertical wins get recognized
        self.game_board.board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", "●", "●", "●", "●", " "]
        ]
        
        # checks if a vertical win gets recognizes
        self.assertTrue(self.game_board.check_winner(PLAYER_ONE_SYMBOL))
        
        # reset board
        self.game_board.reset()
        
        
        ## case three: check if diagonal wins get recognized
        self.game_board.board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", "●", " "],
            [" ", " ", " ", " ", "●", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            [" ", " ", "●", " ", " ", " ", " "]
        ]
        
        # checks if a vertical win gets recognizes
        self.assertTrue(self.game_board.check_winner(PLAYER_ONE_SYMBOL))
        
        # reset board
        self.game_board.reset()


    def testCheckIfBoardIsFullMethod(self):
        '''
        the test method [testCheckIfBoardIsFullMethod] checks if the board class recognizes it
        when the board is full
        '''
        
        # simulate a full board
        self.game_board.board = [
            ["○", "●", "●", "○", "●", "○", "○"],
            ["●", "●", "○", "●", "○", "●", "●"],
            ["○", "●", "●", "○", "○", "●", "○"],
            ["●", "○", "○", "●", "○", "●", "○"],
            ["○", "○", "●", "●", "●", "○", "○"],
            ["●", "○", "●", "●", "○", "●", "●"]
        ]
        
        # checks if the method [is_board_full] works right and returns True
        self.assertTrue(self.game_board.is_board_full())
        
        
        # simulate board with empty space
        self.game_board.board = [
            ["○", "○", "●", " ", "●", "○", "○"],
            ["●", "●", "○", "●", "○", "●", "●"],
            ["○", "●", "●", "○", "●", "●", "○"],
            ["●", "○", "○", "○", "○", "●", "○"],
            ["○", "○", "●", "●", "●", "○", "○"],
            ["●", "○", "●", "●", "○", "●", "●"]
        ]
        
        # checks if the method returns false because of the free space
        self.assertFalse(self.game_board.is_board_full())
        
        # reset board
        self.game_board.reset()
    
    
    def testIsDrawMethdod(self):
        
        # simulate a full board with no winner
        self.game_board.board = [
            ["○", "●", "●", "○", "●", "○", "○"],
            ["●", "●", "○", "●", "○", "●", "●"],
            ["○", "●", "●", "○", "○", "●", "○"],
            ["●", "○", "○", "●", "○", "●", "○"],
            ["○", "○", "●", "●", "●", "○", "○"],
            ["●", "○", "●", "●", "○", "●", "●"]
        ]
        
        # checks if the is_draw method works correctly and returns True
        self.game_board.is_draw()
        
        # reset board
        self.game_board.reset()
        
    
unittest.main()