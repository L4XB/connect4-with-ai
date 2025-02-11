from src.constants import AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL
from src.enviroment.game_board import GameBoard
from src.agents.ml_agent.agent import AIAgent
from src.agents.mini_max_agent import MiniMaxAgent
from src.agents.smart_agent import SmartAgent
from src.agents.random_agent import RandomAgent

def play_game_vs_agent():
    '''
    the method [play_game_vs_agent] gives the opportunity to play as a human against
    an Agent. The agents starts with a move, after this move, the human and the agent pick
    their moves after each other until one of them wins or it ends in a draw.
    '''
    
    # create gameboard
    board = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
    
    # ask which agent to play against
    while True:
        agent_type = input("Which agent do you want to play against? Enter 'ml', 'minimax', 'smart', or 'random': ").strip().lower()
        if agent_type == 'ml':
            agent = AIAgent("src/agents/ml_agent/models/connect4_model_full_trained.pth", PLAYER_ONE_SYMBOL)
            break
        elif agent_type == 'minimax':
            agent = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
            break
        elif agent_type == 'smart':
            agent = SmartAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
            break
        elif agent_type == 'random':
            agent = RandomAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_ONE_SYMBOL)
            break
        else:
            print("Invalid input. Please enter 'ml', 'minimax', 'smart', or 'random'.")
    
    # ask who should start the game
    while True:
        start_player = input("Who should start the game? Enter 'agent' or 'player': ").strip().lower()
        if start_player in ['agent', 'player']:
            break
        else:
            print("Invalid input. Please enter 'agent' or 'player'.")
    
    # game interactions 
    while True:
        if start_player == 'agent':
            # get a move from the agent
            agent_move = agent.get_move(board.board)
            
            # insert the agent move into the board
            board.insert_token(agent_move, agent.symbol)
            
            # print agent move
            print(f"Agent places a Token in Column: {agent_move}")
            
            # draw the board with the inserted move
            board.draw_board()
            
            # check if the agent won with that move
            if(board.check_winner(agent.symbol)):
                print("Agent Won!")
                break
            
            # check if the board is full and the game ends in a draw
            if(board.is_draw()):
                print("It's a draw!")
                break
            
            # switch to player for the next move
            start_player = 'player'
        
        if start_player == 'player':
            # player move
            while True:
                # get player input
                player_move = int(input("Enter your move (column, e.g. '0'): "))
                
                # insert the move, if it's an invalid move the player will be asked to enter another move
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
                print("It's a draw!")
                break
            
            # switch to agent for the next move
            start_player = 'agent'

play_game_vs_agent()