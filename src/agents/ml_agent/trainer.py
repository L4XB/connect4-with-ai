import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from model import Connect4CNN

class Connect4Dataset(Dataset):
    def __init__(self, data_paths):
        self.data = []
        # Eigene Daten laden
        for path in data_paths:
            print(f"Loading data from {path}...")
            with open(path, "rb") as f:
                self.data += pickle.load(f)
            print(f"Loaded {len(self.data)} samples so far.")
        
        # UCI-Daten werden nicht verwendet, da sie keine Zuginformationen enthalten
        print("Skipping UCI dataset as it does not contain move information.")
        
        print(f"Total dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, move = self.data[idx]
        # Konvertiere das Brett in ein 2D-Tensor-Format
        state_tensor = self.board_to_tensor(state)
        return state_tensor, torch.tensor(move, dtype=torch.long)

    def convert_uci_to_board(self, features):
        '''
        Konvertiert UCI-Daten (1D-Array) in das erwartete 2D-Brett-Format.
        '''
        board = []
        for row in range(6):
            board_row = []
            for col in range(7):
                cell = features[row * 7 + col]
                if cell == 0:
                    board_row.append(' ')  # Leeres Feld
                elif cell == 1:
                    board_row.append('●')  # Spieler 1
                elif cell == 2:
                    board_row.append('○')  # Spieler 2
            board.append(board_row)
        return board

    def board_to_tensor(self, board):
        # Ensure the board is in the correct format:
        if not isinstance(board[0], list):  # If board is a flat list (from generated data)
            board = [board[i:i+7] for i in range(0, len(board), 7)]
            # Convert integer values to symbols:
            board = [[' ' if cell == 0 else '●' if cell == 1 else '○' for cell in row] for row in board]

        # Now proceed with the existing conversion:
        channel_self = [[1.0 if cell == '●' else 0.0 for cell in row] for row in board]
        channel_opp = [[1.0 if cell == '○' else 0.0 for cell in row] for row in board]
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
    total_batches = len(loader)
    print(f"Training with {total_batches} batches per epoch")

    print("\nStarting training...")
    # Training
    for epoch in range(80):
        model.train()
        total_loss = 0
        
        for batch_idx, (states, moves) in enumerate(loader):
            states, moves = states.to(device), moves.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, moves)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Print progress at 25%, 50%, 75% of epoch
            if batch_idx in [total_batches // 4, total_batches // 2, 3 * total_batches // 4]:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Epoch {epoch+1:3d}: {progress:3.0f}% complete | Current loss: {loss.item():.4f}")
        
        # Epoch summary
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d} completed | Average loss: {avg_loss:.4f}")
        
        # Print a blank line every 10 epochs for better readability
        if (epoch + 1) % 10 == 0:
            print()

    print("\nTraining completed! Saving model to 'connect4_model.pth'")
    torch.save(model.state_dict(), "connect4_model.pth")

# Start training
train()