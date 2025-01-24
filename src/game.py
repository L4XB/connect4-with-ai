from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from enviroment.game_board import GameBoard
from agents.random_agent import RandomAgent
from agents.smart_agent import SmartAgent

board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)

random_Agent = RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
smart_Agent = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)

print("Smart Agent : " + PLAYER_TWO_SYMBOL)

board.insert_token(0, PLAYER_TWO_SYMBOL)
board.insert_token(1, PLAYER_TWO_SYMBOL)
board.insert_token(2, PLAYER_TWO_SYMBOL)

board.insert_token(4, PLAYER_ONE_SYMBOL)
board.insert_token(5, PLAYER_ONE_SYMBOL)

board.draw_board()

board.insert_token(smart_Agent.get_move(board.board), smart_Agent.symbol)

board.draw_board()
print(board.check_winner(smart_Agent.symbol))



