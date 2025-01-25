from constants import AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

from enviroment.game_board import GameBoard
from agents.mini_max_agent import MiniMaxAgent

def play_game_vs_agent():
    
    board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
    agent = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_ROWS, PLAYER_ONE_SYMBOL)
    
    while True:
        
        agent_move = agent.get_move(board.board)
        board.insert_token(agent_move, agent.symbol)
        board.draw_board()
        
        if(board.check_winner(agent.symbol)):
            print("Agent Won!")
            break
        
        if(board.is_draw()):
            print("Its a draw!")
            break
        
        # player move
        while True:
            player_move = int(input("Enter your move (column, e.g. '0'): "))
            if(not board.insert_token(player_move, agent.opponent_symbol)):
                print("Invalid Move")
            else:
                break
    
        board.draw_board()
        
        if(board.check_winner(agent.opponent_symbol)):
            print("Player Won!")
            break
        
        if(board.is_draw()):
            print("Its a draw!")
            break  


play_game_vs_agent()
