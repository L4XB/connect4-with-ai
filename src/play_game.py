from src.constants import AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

from src.enviroment.game_board import GameBoard
from src.agents.ml_agent.agent import AIAgent

def play_game_vs_agent():
    '''
    the method [play_game_vs_agent] gives the oportunity to play as a humand against
    an Agent. The agents starts with a move, after this move, the humand and the agent pick
    their moves after each other until one of wins or its ends in a draw.
    '''
    
    # create gameboard and agent
    board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
    agent = AIAgent("src/agents/ml_agent/model/connect4_model_good_performance.pth", PLAYER_ONE_SYMBOL)
    
    # game interrations 
    while True:
        
        # get a move from the agent
        agent_move = agent.get_move(board.board)
        
        #insert the agent move into the board
        board.insert_token(agent_move, agent.symbol)
        
        # print aent move
        print(f"Agent places a Token in Column: {agent_move}")
        
        # draw the board with the inserted move
        board.draw_board()
        
        # check if the agent won with that move
        if(board.check_winner(agent.symbol)):
            print("Agent Won!")
            break
        
        # check if the board is full and the game ends in a draw
        if(board.is_draw()):
            print("Its a draw!")
            break
        
        # player move
        while True:
            
            # get player input
            player_move = int(input("Enter your move (column, e.g. '0'): "))
            
            # insert the move, if its an invalid move the player will be asked to enter a other move
            if(not board.insert_token(player_move, agent.opponent_symbol)):
                print("Invalid Move")
            else:
                print(f"Player places a Token in Column: {player_move}")
                break
        
        
        # draw board again after the player move is inserted
        board.draw_board()
        
        # check if the player won with that move
        if(board.check_winner(agent.opponent_symbol)):
            print("Player Won!")
            break
        
        # check if the board is full and the game ends in a draw
        if(board.is_draw()):
            print("Its a draw!")
            break  


play_game_vs_agent()
