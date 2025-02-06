import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from src.agents.ml_agent.agent import AIAgent
from src.constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS

class TestAIAgentMethods(unittest.TestCase):
    
    def __init__(self, methodName="runAIAgentTests"):
        super().__init__(methodName)
        
        # sets the model path for the tests
        self.model_path = "src/agents/ml_agent/models/connect4_model_good_performance.pth"
        
        # assigns the symbols
        self.symbol = PLAYER_ONE_SYMBOL
        self.opponent_symbol = PLAYER_TWO_SYMBOL
        
        # creates a AiAgent object with the model path to test it
        self.agent = AIAgent(self.model_path, self.symbol)


    def testInitialization(self):
        """the tests method [testInitialization] tests if the setup in the [__init__] was succesfull"""
        
        # checks every assertion from the [__init__]
        self.assertEqual(self.agent.symbol, self.symbol)
        self.assertEqual(self.agent.opponent_symbol, self.opponent_symbol)
        self.assertEqual(self.agent.cols, AMOUNT_COLUMNS)
        self.assertEqual(self.agent.rows, AMOUNT_ROWS)
        self.assertTrue(self.agent.model.training is False)


    def testBoardToTensorConversion(self):
        """the test method [testBoardToTensorConversion] checks the board to tensor conversion with different 
        board states during the game"""
        
        # test with empty board
        empty_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        
        # coverts the empty_board to a tensore
        tensor = self.agent.board_to_tensor(empty_board)
        
        # tests the shae of the tensor
        self.assertEqual(tensor.shape, (1, 2, AMOUNT_ROWS, AMOUNT_COLUMNS))
        
        # checks if all elements in the tensor a 0
        self.assertTrue((tensor == 0).all())

        # test board with some random moves
        test_board = [
            [" ", " ", " ", "○", " ", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            [" ", " ", " ", "○", " ", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            [" ", " ", " ", "○", " ", " ", " "]
        ]
        
        # converts the test_board to a tensor
        tensor = self.agent.board_to_tensor(test_board)
        
        # checks if the tensor shape is correct
        self.assertEqual(tensor.shape, (1, 2, AMOUNT_ROWS, AMOUNT_COLUMNS))


    def testPossibleMovesDetection(self):
        """the test method [testPossibleMovesDetection] checks the detection of available columns"""
        
        # test with empty board
        empty_board = [[" "]*AMOUNT_COLUMNS for _ in range(AMOUNT_ROWS)]
        moves = self.agent._get_possible_moves(empty_board)
        self.assertEqual(moves, list(range(AMOUNT_COLUMNS)))

        # test with partially filled board
        test_board = [
            [" ", " ", " ", "○", " ", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            [" ", " ", " ", "○", " ", " ", " "],
            [" ", " ", " ", "●", " ", " ", " "],
            ["●", "○", " ", "○", "●", " ", " "]
        ]
        moves = self.agent._get_possible_moves(test_board)
        self.assertEqual(moves, [0,1,2,4,5,6])


    def testMovePlayingMechanics(self):
        """the test method [testMovePlayingMechanics] checks the token placement mechanics"""
        
        test_board = [[" "]*AMOUNT_COLUMNS for _ in range(AMOUNT_ROWS)]
        
        # test valid move
        new_board = self.agent._play_move(test_board, 3, self.symbol)
        self.assertEqual(new_board[5][3], self.symbol)
        
        # test filling column
        for _ in range(AMOUNT_ROWS):
            new_board = self.agent._play_move(new_board, 3, self.symbol)
        self.assertEqual(new_board[0][3], self.symbol)
        
        # test invalid column
        invalid_board = self.agent._play_move(test_board, -1, self.symbol)
        self.assertFalse(invalid_board)


    @patch('torch.softmax')
    def testModelPredictionFallback(self, mock_softmax):
        """the test method [testModelPredictionFallback] checks the model prediction when no immediate moves available"""
        
        # mock model output
        mock_softmax.return_value = torch.tensor([0.1, 0.5, 0.4])
        self.agent.model = MagicMock()
        self.agent.model.return_value = torch.tensor([[1, 2, 3]])

        test_board = [
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", "●", "○", " ", " ", " "],
            [" ", " ", "●", "○", " ", " ", " "],
            [" ", " ", "●", "○", " ", " ", " "]
        ]
        
        move = self.agent.get_move(test_board)
        self.assertIn(move, [0,1,2,3,4,5,6])


    def testFullColumnHandling(self):
        """the test method [testFullColumnHandling] checks the behavior when trying to play in full column"""
        
        full_col_board = [
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "],
            ["●", " ", " ", " ", " ", " ", " "]
        ]
        moves = self.agent._get_possible_moves(full_col_board)
        self.assertNotIn(0, moves)


unittest.main()