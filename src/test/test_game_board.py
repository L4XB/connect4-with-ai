import unittest
from src.enviroment.game_board import GameBoard
from src.constants import AMOUNT_COLUMNS, AMOUNT_ROWS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class TestGameBoardMethods(unittest.TestCase):
    
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.game_board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
    
    def testInitialBoardSetup(self):
        default_board_layout = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        
        self.assertEqual(default_board_layout, self.game_board.board)
    
    
    def testRestBoardMethod(self):
        
        self.game_board.board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        
        default_board_layout = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        
        self.game_board.reset()
        self.assertEqual(self.game_board.board, default_board_layout)
    
    
    def testInsertTokenMethodWhenInsertionIsPossible(self):
        
        self.assertTrue(self.game_board.insert_token(0, PLAYER_ONE_SYMBOL))
        
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
        





unittest.main()