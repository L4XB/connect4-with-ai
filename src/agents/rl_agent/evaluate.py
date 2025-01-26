import numpy as np
from tqdm import tqdm
from agents.rl_agent.agent import RLAgent
from agents.mini_max_agent import MiniMaxAgent
from enviroment.game_board import GameBoard
from constants import *

def evaluate(num_games=100):
    # Initialisierung
    rl_agent = RLAgent(PLAYER_ONE_SYMBOL)
    rl_agent.load_model()
    minimax = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL, max_depth=4)
    
    results = {'RL_Wins': 0, 'MiniMax_Wins': 0, 'Draws': 0}
    
    # Evaluationsloop
    for _ in tqdm(range(num_games), desc="Evaluation"):
        env = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
        done = False
        turn = 0  # Abwechselnder Startspieler
        
        while not done:
            if turn % 2 == 0:
                # RL Agent
                valid_moves = [c for c in range(AMOUNT_COLUMNS) if env.board[0][c] == ' ']
                action = rl_agent.get_action(env.board, valid_moves, training=False)
                env.insert_token(action, PLAYER_ONE_SYMBOL)
                if env.check_winner(PLAYER_ONE_SYMBOL):
                    results['RL_Wins'] += 1
                    done = True
            else:
                # MiniMax Agent
                action = minimax.get_move(env.board)
                env.insert_token(action, PLAYER_TWO_SYMBOL)
                if env.check_winner(PLAYER_TWO_SYMBOL):
                    results['MiniMax_Wins'] += 1
                    done = True
            
            if env.is_draw():
                results['Draws'] += 1
                done = True
            
            turn += 1
    
    # Ergebnisse anzeigen
    print("\nEvaluationsergebnisse:")
    print(f"RL Siege: {results['RL_Wins']} ({results['RL_Wins']/num_games*100:.1f}%)")
    print(f"MiniMax Siege: {results['MiniMax_Wins']} ({results['MiniMax_Wins']/num_games*100:.1f}%)")
    print(f"Unentschieden: {results['Draws']} ({results['Draws']/num_games*100:.1f}%)")


evaluate()