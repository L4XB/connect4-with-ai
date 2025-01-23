from .constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from src.env.game_board import GameBoard

board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)

board.insert_token(3, 'X')
board.insert_token(3, 'O')
board.insert_token(0, 'X')
board.insert_token(6, 'O')


board.draw_board()