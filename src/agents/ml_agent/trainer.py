import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Connect4CNN
from constants import PLAYER_ONE_SYMBOL, PLAYER_TWO_SYMBOL

class Connect4Dataset(Dataset):
    def __init__(self, data_paths):
        self.data = []
        for path in data_paths:
            print(f"Loading data from {path}...")
            with open(path, "rb") as f:
                self.data += pickle.load(f)
            print(f"Loaded {len(self.data)} samples so far.")
        print("Skipping UCI dataset as it does not contain move information.")
        
        print(f"Total dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, move = self.data[idx]
        state_tensor = self.board_to_tensor(state)
        return state_tensor, torch.tensor(move, dtype=torch.long)

    def convert_uci_to_board(self, features):
        board = []
        for row in range(6):
            board_row = []
            for col in range(7):
                cell = features[row * 7 + col]
                if cell == 0:
                    board_row.append(' ')
                elif cell == 1:
                    board_row.append(PLAYER_ONE_SYMBOL)
                elif cell == 2:
                    board_row.append(PLAYER_TWO_SYMBOL)
            board.append(board_row)
        return board

    def board_to_tensor(self, board):
        if not isinstance(board[0], list):
            board = [board[i:i+7] for i in range(0, len(board), 7)]
            board = [[' ' if cell == 0 else PLAYER_ONE_SYMBOL if cell == 1 else PLAYER_TWO_SYMBOL for cell in row] for row in board]

        channel_self = [[1.0 if cell == PLAYER_ONE_SYMBOL else 0.0 for cell in row] for row in board]
        channel_opp = [[1.0 if cell == PLAYER_TWO_SYMBOL else 0.0 for cell in row] for row in board]
        return torch.FloatTensor([channel_self, channel_opp])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = Connect4CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    # Create dataset and DataLoader
    dataset = Connect4Dataset([
        "connect4_data_3d_10000g.pkl",
        "connect4_data_3d_5000g.pkl"
    ])
    print(f"Dataset loaded with {len(dataset):,} samples")

    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # For tracking loss
    epoch_losses = []
    
    print("\nStarting training...")
    # Training
    for epoch in range(50):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1:3}/{50}", unit="batch", leave=False)

        for states, moves in progress_bar:
            states, moves = states.to(device), moves.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, moves)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            current_loss = loss.item()
            total_loss += current_loss
            
            # Update progress bar
            progress_bar.set_postfix_str(f"loss: {current_loss:.4f}")

        # Epoch statistics
        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        
        # Update main progress bar description
        print(f"Epoch {epoch+1:3d} completed | Average loss: {avg_loss:.4f}")

    # Save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, 'b-o', linewidth=2)
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()

    print("\nTraining completed! Saving model to 'connect4_model.pth'")
    torch.save(model.state_dict(), "connect4_model.pth")

train()