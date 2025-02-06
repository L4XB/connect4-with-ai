import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.agents.ml_agent.model import Connect4CNN
from src.agents.ml_agent.pipe.data_loader import Connect4Dataset

def train():
    '''
    the method [train] initializes and trains the Connect4CNN model using the specified dataset.
    It includes data loading, model training, validation, and early stopping.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # initialize model
    model = Connect4CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # create dataset and DataLoader
    dataset = Connect4Dataset([
        "src/agents/ml_agent/data/connect4_data_minimax_3d_750g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_3d_800g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_3d_vs_minimax_4d_1200g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_3d_vs_minimax_4d_150g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_3d_vs_minimax_4d_900g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_250g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_400g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_450g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_500g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_550g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_800g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_vs_minimax_5d_100g.pkl",
        "src/agents/ml_agent/data/connect4_data_minimax_4d_vs_minimax_5d_150g.pkl",
        #"src/agents/ml_agent/data/connect4_data_smart_2000g.pkl",
        #"src/agents/ml_agent/data/connect4_data_smart_vs_minimax_3d_1000g.pkl",
        #"src/agents/ml_agent/data/connect4_data_smart_vs_minimax_4d_1000g.pkl",
    ])
    print(f"Dataset loaded with {len(dataset):,} samples")
    sample_state, sample_move = dataset[0]
    print("Sample state shape:", sample_state.shape)
    print("Sample move:", sample_move)

    # split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # for tracking loss and early stopping
    epoch_losses = []
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    print("\nStarting training...")
    
    # training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3}/{10}", unit="batch", leave=False)

        for states, moves in progress_bar:
            states, moves = states.to(device), moves.to(device)

            # forward pass
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, moves)

            # backward pass
            loss.backward()
            optimizer.step()

            # update metrics
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix_str(f"loss: {current_loss:.4f}")

        # calculate average training loss
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for states, moves in val_loader:
                states, moves = states.to(device), moves.to(device)
                outputs = model(states)
                loss = criterion(outputs, moves)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # epoch statistics
        print(f"Epoch {epoch+1:3d} completed | Train loss: {avg_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        # early stopping check
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

    # save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, 'b-o', linewidth=2)
    plt.title("Training Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()

    print("\nTraining completed!")

train()