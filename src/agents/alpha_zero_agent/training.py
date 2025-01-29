import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from agents.alpha_zero_agent.mcts import MCTS
from constants import *
import torch.nn.functional as F
from tqdm import tqdm

def self_play(model, num_games=200, num_simulations=200):
    mcts = MCTS(model, 6, 7, num_simulations)
    data = []
    
    for game_idx in tqdm(range(num_games), desc="Self-play games"):
        board = [[' ']*7 for _ in range(6)]
        history = []
        player = PLAYER_ONE_SYMBOL
        
        while True:
            col = mcts.run(board, player)
            valid_moves = mcts.get_valid_moves(board)
            
            # Play move
            for row in reversed(range(6)):
                if board[row][col] == ' ':
                    board[row][col] = player
                    break
            
            # Record state and policy
            state = mcts.board_to_state(board, player)
            visits = np.array([mcts.root.children[c].visit_count if c in mcts.root.children else 0 for c in range(7)])
            policy = visits / visits.sum()
            
            history.append((state, policy, player))
            
            # Check terminal
            winner = mcts.check_winner(board)
            if winner or not valid_moves.any():
                # Assign values
                value = 0
                if winner:
                    value = 1 if winner == history[0][2] else -1
                for s, p, pl in history:
                    data.append((s, p, value if pl == history[0][2] else -value))
                break
                
            player = mcts.get_opponent(player)
    
    print(f"Collected {len(data)} training examples from {num_games} games")
    return data

def train(model, data, epochs=10, batch_size=128, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    states = torch.stack([d[0] for d in data])
    policies = torch.tensor([d[1] for d in data], dtype=torch.float32)
    values = torch.tensor([d[2] for d in data], dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        total_loss = 0
        for batch in loader:
            s, p, v = batch
            optimizer.zero_grad()
            p_pred, v_pred = model(s)
            
            loss_p = F.kl_div(p_pred, p, reduction='batchmean')
            loss_v = F.mse_loss(v_pred, v)
            loss = loss_p + loss_v
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
    
    return model