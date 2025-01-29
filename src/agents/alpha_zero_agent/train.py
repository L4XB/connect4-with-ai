from agents.alpha_zero_agent.agent import AlphaZeroAgent, MCTS
from agents.alpha_zero_agent.training import self_play, train
from agents.mini_max_agent import MiniMaxAgent
from agents.alpha_zero_agent.evaluate import evaluate_against_minimax
import torch
from constants import *

# Hyperparameter
NUM_ITERATIONS = 10
NUM_SELF_PLAY = 500
NUM_EPOCHS = 20
EVALUATION_GAMES = 100  # Anzahl der Spiele zur Evaluation
WINRATE_THRESHOLD = 0.8  # Winrate, bei der das Training gestoppt wird

# Initialisierung
agent = AlphaZeroAgent(6, 7, PLAYER_ONE_SYMBOL)
mcts = MCTS(agent.model, 6, 7)
mini_max_agent = MiniMaxAgent(6, 7, PLAYER_TWO_SYMBOL, max_depth=2)  # Start mit depth=2

for iteration in range(NUM_ITERATIONS):
    print(f"Iteration {iteration+1}/{NUM_ITERATIONS}")
    
    # Self-Play
    print("Running self-play...")
    data = self_play(agent.model, mcts, NUM_SELF_PLAY)
    
    # Training
    print("Training model...")
    agent.model = train(agent.model, data, NUM_EPOCHS)
    
    # Speichern des Modells
    model_path = f"az_model_iter_{iteration}.pth"
    torch.save(agent.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluation gegen MiniMax
    print("Evaluating against MiniMax...")
    winrate = evaluate_against_minimax(agent, mini_max_agent, EVALUATION_GAMES)
    print(f"Winrate against MiniMax (depth={mini_max_agent.max_depth}): {winrate * 100:.2f}%")
    
    # Dynamische Anpassung der MiniMax-Tiefe
    if winrate > 0.7:
        mini_max_agent.max_depth = 4
    elif winrate > 0.5:
        mini_max_agent.max_depth = 3
    else:
        mini_max_agent.max_depth = 2
    
    # Stoppen, wenn Winrate über dem Schwellenwert liegt
    if winrate >= WINRATE_THRESHOLD:
        print(f"Winrate exceeded {WINRATE_THRESHOLD * 100}%. Stopping training.")
        break