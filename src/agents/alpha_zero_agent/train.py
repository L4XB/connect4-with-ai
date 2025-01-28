from constants import *
from agents.alpha_zero_agent.agent import AlphaZeroAgent
from agents.mini_max_agent import MiniMaxAgent
from agents.alpha_zero_agent.mcts import MCTS
import torch
from agents.alpha_zero_agent.training import self_play, train
from constants import *


# Initialisierung der Agenten
alpha_zero_agent = AlphaZeroAgent(rows= AMOUNT_ROWS, cols= AMOUNT_COLUMNS, symbol=PLAYER_ONE_SYMBOL)

# Self-Play durchführen
mcts = MCTS(alpha_zero_agent.model, rows=6, cols=7, num_simulations=100)
training_data = self_play(alpha_zero_agent.model, mcts, num_games=100)

# Training des Modells
optimizer = torch.optim.Adam(alpha_zero_agent.model.parameters(), lr=0.001)
alpha_zero_agent.model = train(alpha_zero_agent.model, optimizer, training_data, epochs=10)

# Modell speichern
torch.save(alpha_zero_agent.model.state_dict(), "alpha_zero_model.pth")