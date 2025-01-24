from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from enviroment.game_board import GameBoard
from agents.random_agent import RandomAgent
from agents.smart_agent import SmartAgent

board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)

random_Agent = RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
smart_Agent = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)



random_agent_move = random_Agent.get_move(board.board)
board.insert_token(random_agent_move, random_Agent.symbol)

smart_agent_move = smart_Agent.get_move(board.board)
board.insert_token(smart_agent_move, smart_Agent.symbol)


board.draw_board()
