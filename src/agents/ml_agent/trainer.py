import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from model import Connect4CNN

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
                    board_row.append('●')
                elif cell == 2:
                    board_row.append('○')
            board.append(board_row)
        return board

    def board_to_tensor(self, board):
        if not isinstance(board[0], list):
            board = [board[i:i+7] for i in range(0, len(board), 7)]
            board = [[' ' if cell == 0 else '●' if cell == 1 else '○' for cell in row] for row in board]

        channel_self = [[1.0 if cell == '●' else 0.0 for cell in row] for row in board]
        channel_opp = [[1.0 if cell == '○' else 0.0 for cell in row] for row in board]
        return torch.FloatTensor([channel_self, channel_opp])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Connect4CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = Connect4Dataset([
        # add datasets
    ])
    print(f"Dataset loaded with {len(dataset):,} samples")
    
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    total_batches = len(loader)
    print(f"Training with {total_batches} batches per epoch")

    print("\nStarting training...")
    for epoch in range(80):
        model.train()
        total_loss = 0
        
        for batch_idx, (states, moves) in enumerate(loader):
            states, moves = states.to(device), moves.to(device)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, moves)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if batch_idx in [total_batches // 4, total_batches // 2, 3 * total_batches // 4]:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Epoch {epoch+1:3d}: {progress:3.0f}% complete | Current loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d} completed | Average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            print()

    print("\nTraining completed! Saving model to 'connect4_model.pth'")
    torch.save(model.state_dict(), "connect4_model.pth")

train()