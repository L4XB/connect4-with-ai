from torch.utils.data import DataLoader, TensorDataset
from constants import *
import torch.nn.functional as F
import torch
import numpy as np

def self_play(model, mcts, num_games):
    training_data = []
    print(f"Starting self-play for {num_games} games...")
    
    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}")
        board = [[' ' for _ in range(mcts.cols)] for _ in range(mcts.rows)]
        history = []
        player = PLAYER_ONE_SYMBOL
        game_done = False
        
        while not game_done:
            action = mcts.run(board, player)
            valid_moves = mcts.get_valid_moves(board)
            state = mcts.board_to_state(board, player)
            
            visit_counts = np.array([mcts.root.children[col].visit_count if col in mcts.root.children else 0 for col in range(mcts.cols)])
            policy = visit_counts / visit_counts.sum()
            
            history.append((state, policy, player))
            board = mcts.play_move(board, action, player)
            
            winner = mcts.check_winner(board)
            if winner or not valid_moves.any():
                for idx, (state, policy, p) in enumerate(history):
                    value = 1 if p == winner else -1 if winner else 0
                    training_data.append((state, policy, value))
                game_done = True
            
            # Switch players
            player = mcts.get_opponent(player)
    
    print(f"Self-play completed. Collected {len(training_data)} data points.")
    return training_data

def train(model, optimizer, data, epochs=10, batch_size=32):
    states, policies, values = zip(*data)
    states = torch.stack(states)
    policies = torch.tensor(np.array(policies))
    values = torch.tensor(np.array(values)).float().unsqueeze(1)
    
    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    print(f"Training started for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        
        for batch_num, (s, p, v) in enumerate(loader):
            optimizer.zero_grad()
            p_pred, v_pred = model(s)
            
            # Compute the losses
            loss_policy = F.kl_div(p_pred, p, reduction='batchmean')
            loss_value = F.mse_loss(v_pred, v)
            loss = loss_policy + loss_value
            
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            if batch_num % 10 == 0:  # Print every 10th batch
                print(f"Batch {batch_num}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch + 1} completed. Total loss: {total_loss:.4f}")
    
    print(f"Training completed.")
    return model
