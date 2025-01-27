import numpy as np
import torch
from tqdm import tqdm
from agents.rl_agent.agent import RLAgent
from agents.mini_max_agent import MiniMaxAgent
from enviroment.game_board import GameBoard
from constants import *

def evaluate_agent(current_agent, opponent, num_eval_episodes=50):
    """Evaluates the current agent against the MiniMax opponent without exploration."""
    wins = 0
    original_epsilon = current_agent.epsilon  # <-- Originalwert zwischenspeichern
    current_agent.epsilon = 0.0

    for _ in range(num_eval_episodes):
        env = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
        state = env.reset()
        done = False
        
        while not done:
            # RL Agent's Zug
            valid_moves = [c for c in range(AMOUNT_COLUMNS) if env.board[0][c] == ' ']
            action = current_agent.get_action(env.board, valid_moves)
            env.insert_token(action, PLAYER_ONE_SYMBOL)
            
            if env.check_winner(PLAYER_ONE_SYMBOL):
                wins += 1
                done = True
                break
            elif env.is_draw():
                done = True
                break
            
            # MiniMax Agent's Zug
            mm_action = opponent.get_move(env.board)
            env.insert_token(mm_action, PLAYER_TWO_SYMBOL)
            
            if env.check_winner(PLAYER_TWO_SYMBOL):
                done = True
            elif env.is_draw():
                done = True

    current_agent.epsilon = original_epsilon  # <-- Zurück zum ursprünglichen Wert
    win_rate = wins / num_eval_episodes
    return win_rate

def train():
    env = GameBoard(AMOUNT_ROWS, AMOUNT_COLUMNS)
    rl_agent = RLAgent(PLAYER_ONE_SYMBOL)
    minimax = MiniMaxAgent(AMOUNT_ROWS, AMOUNT_COLUMNS, PLAYER_TWO_SYMBOL, max_depth=4)
    
    episodes = 500
    eval_interval = 50  # Führe Evaluation alle 50 Episoden durch
    best_win_rate = -np.inf
    
    # Training Loop
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        done = False
        
        while not done:
            # RL Agent's Zug
            valid_moves = [c for c in range(AMOUNT_COLUMNS) if env.board[0][c] == ' ']
            action = rl_agent.get_action(env.board, valid_moves)
            prev_state = np.copy(env.board)
            env.insert_token(action, PLAYER_ONE_SYMBOL)
            
            # Reward Berechnung
            reward = 0
            if env.check_winner(PLAYER_ONE_SYMBOL):
                reward = 1
                done = True
            elif env.is_draw():
                reward = 0.3
                done = True
            else:
                # MiniMax Agent's Zug
                mm_action = minimax.get_move(env.board)
                env.insert_token(mm_action, PLAYER_TWO_SYMBOL)
                
                if env.check_winner(PLAYER_TWO_SYMBOL):
                    reward = -1
                    done = True
                elif env.is_draw():
                    reward = 0.1
                    done = True
            
            # Experience speichern und lernen
            next_state = np.copy(env.board)
            rl_agent.remember(prev_state, action, reward, next_state, done)
            rl_agent.learn()
        
        # Epsilon Decay
        rl_agent.decay_epsilon()
        
        # Evaluierung und Target-Netzwerk Update
        if episode % eval_interval == 0:
            current_win_rate = evaluate_agent(rl_agent, minimax)
            
            # Update Target-Netzwerk nur bei Verbesserung
            if current_win_rate > best_win_rate:
                rl_agent.update_target_net()
                best_win_rate = current_win_rate
                rl_agent.save_model()  # Speichere das beste Modell
                print(f"Neues bestes Modell mit Win Rate: {current_win_rate:.2f}")
    
    print("Training abgeschlossen. Bestes Modell gespeichert.")

train()