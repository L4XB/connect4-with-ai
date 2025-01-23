from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from enviroment.game_board import GameBoard

board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)

board.insert_token(3, PLAYER_ONE_SYMBOL)
board.insert_token(3, PLAYER_TWO_SYMBOL)
board.insert_token(0, PLAYER_ONE_SYMBOL)
board.insert_token(6, PLAYER_TWO_SYMBOL)


board.draw_board()