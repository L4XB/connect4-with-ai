from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL, AMOUNT_COLUMNS, AMOUNT_ROWS
from enviroment.game_board import GameBoard
from agents.smart_agent import SmartAgent
from agents.mini_max_agent import MiniMaxAgent 

def simulate_games_minimax_vs_smart(num_games):
    '''
    the method [simulate_games_minimax_vs_smart] simulates a connect4 game between the smart and 
    the minimax agent. 
    The parameter [num_games] can be used to set the amount of the games the agents play against
    each other
    '''
    
    minimax_wins = 0
    smart_wins = 0
    draws = 0

    for game in range(num_games):
        board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
        agents = {
            PLAYER_ONE_SYMBOL: MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL),
            PLAYER_TWO_SYMBOL: SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL)
        }
        
        current_player = PLAYER_ONE_SYMBOL
        
        while True:
            # make move
            move = agents[current_player].get_move(board.board)
            board.insert_token(move, current_player)
            
            # check if somone winns
            if board.check_winner(current_player):
                if current_player == PLAYER_ONE_SYMBOL:
                    minimax_wins += 1
                else:
                    smart_wins += 1
                break
            
            # check if its a draw
            if board.is_draw():
                draws += 1
                break
            
            # change player
            current_player = PLAYER_TWO_SYMBOL if current_player == PLAYER_ONE_SYMBOL else PLAYER_ONE_SYMBOL

    # print result
    print(f"Ergebnis nach {num_games} Spielen:")
    print(f"Minimax Agent ({PLAYER_ONE_SYMBOL}) Siege: {minimax_wins}")
    print(f"Smart Agent ({PLAYER_TWO_SYMBOL}) Siege: {smart_wins}")
    print(f"Unentschieden: {draws}")
