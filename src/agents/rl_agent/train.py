import numpy as np
import torch
from tqdm import tqdm
from agents.rl_agent.agent import RLAgent
from agents.mini_max_agent import MiniMaxAgent
from enviroment.game_board import GameBoard
from constants import *

def train():
    # Initialisierung
    env = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
    rl_agent = RLAgent(PLAYER_ONE_SYMBOL)
    minimax = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL, max_depth=4)
    
    # Training Loop
    episodes = 500
    wins = 0
    losses = 0
    draws = 0
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        done = False
        
        while not done:
            # RL Agent Zug
            valid_moves = [c for c in range(AMOUNT_COLUMNS) if env.board[0][c] == ' ']
            action = rl_agent.get_action(env.board, valid_moves)
            prev_state = np.copy(env.board)
            env.insert_token(action, PLAYER_ONE_SYMBOL)
            
            # Reward Berechnung
            reward = 0
            if env.check_winner(PLAYER_ONE_SYMBOL):
                reward = 1
                done = True
                wins += 1
            elif env.is_draw():
                reward = 0.3
                done = True
                draws += 1
            else:
                # MiniMax Agent Zug
                mm_action = minimax.get_move(env.board)
                env.insert_token(mm_action, PLAYER_TWO_SYMBOL)
                
                if env.check_winner(PLAYER_TWO_SYMBOL):
                    reward = -1
                    done = True
                    losses += 1
                elif env.is_draw():
                    reward = 0.1
                    done = True
                    draws += 1
            
            # Experience speichern
            next_state = np.copy(env.board)
            rl_agent.remember(prev_state, action, reward, next_state, done)
            rl_agent.learn()
        
        # Epsilon Decay
        rl_agent.decay_epsilon()
        
        # Target Network Update
        if episode % 50 == 0:
            rl_agent.update_target_net()
        
        # Fortschrittsanzeige
        if episode % 100 == 0:
            print(f"Episode {episode}: Wins={wins}, Losses={losses}, Draws={draws}")
            wins, losses, draws = 0, 0, 0
            
    print(f"Episode {episode}: Wins={wins}, Losses={losses}, Draws={draws}")
    # Modell speichern
    rl_agent.save_model()
    print("Training abgeschlossen. Modell gespeichert.")

train()