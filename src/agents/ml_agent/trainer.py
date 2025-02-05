import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents.ml_agent.model import Connect4CNN

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

    def board_to_tensor(self, state):
        # Handle both 2D and flattened state formats
        if not isinstance(state[0], list):  # Flattened state (e.g., from generated data)
            state = [state[i:i+7] for i in range(0, len(state), 7)]

        # Convert numerical values to channels
        channel_self = [[1.0 if cell == 1 else 0.0 for cell in row] for row in state]
        channel_opp = [[1.0 if cell == -1 else 0.0 for cell in row] for row in state]
        return torch.FloatTensor([channel_self, channel_opp])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = Connect4CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Create dataset and DataLoader
    dataset = Connect4Dataset([
        "connect4_data_minimax_3d_750g.pkl",
        "connect4_data_minimax_3d_800g.pkl",
        "connect4_data_minimax_3d_vs_minimax_4d_1200g.pkl",
        "connect4_data_minimax_3d_vs_minimax_4d_150g.pkl",
        "connect4_data_minimax_3d_vs_minimax_4d_900g.pkl",
        "connect4_data_minimax_4d_250g.pkl",
        "connect4_data_minimax_4d_400g.pkl",
        "connect4_data_minimax_4d_450g.pkl",
        "connect4_data_minimax_4d_500g.pkl",
        "connect4_data_minimax_4d_550g.pkl",
        "connect4_data_minimax_4d_800g.pkl",
        "connect4_data_minimax_4d_vs_minimax_5d_100g.pkl",
        "connect4_data_minimax_4d_vs_minimax_5d_150g.pkl",
        #"connect4_data_smart_2000g.pkl",
        #"connect4_data_smart_vs_minimax_3d_1000g.pkl",
        #"connect4_data_smart_vs_minimax_4d_1000g.pkl",
    ])
    print(f"Dataset loaded with {len(dataset):,} samples")
    sample_state, sample_move = dataset[0]
    print("Sample state shape:", sample_state.shape)
    print("Sample move:", sample_move)

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # For tracking loss and early stopping
    epoch_losses = []
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    print("\nStarting training...")
    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3}/{10}", unit="batch", leave=False)

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
            progress_bar.set_postfix_str(f"loss: {current_loss:.4f}")

        # Calculate average training loss
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for states, moves in val_loader:
                states, moves = states.to(device), moves.to(device)
                outputs = model(states)
                loss = criterion(outputs, moves)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Epoch statistics
        print(f"Epoch {epoch+1:3d} completed | Train loss: {avg_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "connect4_model.pth")
            print(f"Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
        scheduler.step()

    # Save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, 'b-o', linewidth=2)
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()

    print("\nTraining completed!")

# Start training
train()