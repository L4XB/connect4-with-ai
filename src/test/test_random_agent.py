import unittest
from src.enviroment.game_board import GameBoard
from src.agents.random_agent import RandomAgent
from src.constants import AMOUNT_COLUMNS, AMOUNT_ROWS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class TestRandomAgentMethods(unittest.TestCase):
    
    def __init__(self, methodName = "runRandomAgentTest"):
        super().__init__(methodName)
        
        # create a GameBoard instance and safes it as attribut
        self.game_board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
        
        # create a RandomAgent instance and safes it as attributs
        self.random_agent = RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
        
    
    def testGetPossibleColumnsMethod(self):
        '''
        the test method [testGetPossibleColumnsMethod] checks if the reandom agent gets
        the right possible columns and returns them.
        '''
        
        # demo board with two full columns
        self.game_board.board = [
            ["●", " ", " ", "●", " ", " ", " "],
            ["●", " ", " ", "●", " ", " ", " "],
            ["●", " ", " ", "●", " ", " ", " "],
            ["●", " ", " ", "●", " ", " ", " "],
            ["●", " ", " ", "●", " ", " ", " "],
            ["●", " ", " ", "●", " ", " ", " "]
        ]
        
        # call method to get possible moves
        possible_columns_to_evaluate = self.random_agent._get_possible_moves(self.game_board.board)
        
        # check if the output of the methd matches with the reality
        self.assertEqual(possible_columns_to_evaluate, [1, 2, 4, 5, 6])
        
        # reset board
        self.game_board.reset()


unittest.main()