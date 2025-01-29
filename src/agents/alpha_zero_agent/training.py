import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from constants import *
import torch.nn.functional as F
import time

def self_play(model, mcts, num_games=500):
    print(f"Starting self-play with {num_games} games...")
    start_time = time.time()
    training_data = []
    total_moves = 0
    
    for game_num in range(num_games):
        board = [[' ']*mcts.cols for _ in range(mcts.rows)]
        history = []
        player = PLAYER_ONE_SYMBOL
        moves_in_game = 0
        
        while True:
            col = mcts.run(board, player)
            row = next(r for r in reversed(range(mcts.rows)) if board[r][col] == ' ')
            board[row][col] = player
            moves_in_game += 1
            
            state = mcts.board_to_state(board, player)
            visit_counts = np.array([mcts.root.children[c].visit_count if c in mcts.root.children else 0 for c in range(mcts.cols)])
            policy = visit_counts / visit_counts.sum()
            
            history.append((state, policy, player))
            
            winner = mcts.check_winner(board)
            if winner or all(cell != ' ' for row in board for cell in row):
                for state, policy, p in history:
                    value = 1 if p == winner else -1 if winner else 0
                    training_data.append((state, policy, value))
                break
            player = mcts.get_opponent(player)
        
        total_moves += moves_in_game
        if (game_num + 1) % 10 == 0: 
            elapsed = time.time() - start_time
            avg_moves = total_moves / (game_num + 1)
            print(f"Game {game_num + 1}/{num_games} completed. Moves: {moves_in_game}, Avg moves/game: {avg_moves:.1f}, Time: {elapsed:.1f}s")
    
    print(f"\nSelf-play completed in {time.time() - start_time:.1f} seconds")
    print(f"Total positions generated: {len(training_data)}")
    print(f"Average moves per game: {total_moves/num_games:.1f}")
    return training_data

def train(model, data, epochs=20, batch_size=128):
    print(f"\nStarting training with {len(data)} positions")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    start_time = time.time()
    
    states, policies, values = zip(*data)
    states = torch.stack(states)
    policies = torch.tensor(np.array(policies), dtype=torch.float32)
    values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(loader)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_policy_loss = 0
        epoch_value_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            s, p, v = batch
            optimizer.zero_grad()
            p_pred, v_pred = model(s)
            
            loss_p = -torch.sum(p * p_pred) / p.size(0)
            loss_v = F.mse_loss(v_pred, v)
            loss = loss_p + loss_v
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_policy_loss += loss_p.item()
            epoch_value_loss += loss_v.item()
            
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{total_batches}, "
                      f"Loss: {loss.item():.4f} (P: {loss_p.item():.4f}, V: {loss_v.item():.4f})")
        
        # Print epoch summary
        avg_loss = epoch_loss / len(loader)
        avg_policy_loss = epoch_policy_loss / len(loader)
        avg_value_loss = epoch_value_loss / len(loader)
        print(f"\nEpoch {epoch + 1}/{epochs} completed. "
              f"Average Loss: {avg_loss:.4f} (P: {avg_policy_loss:.4f}, V: {avg_value_loss:.4f})")
    
    print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")
    return model