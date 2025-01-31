import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from src.agents.ml_agent.model import Connect4MLP
import numpy as np
from ucimlrepo import fetch_ucirepo 

class Connect4Dataset(Dataset):
    def __init__(self, data_paths):
        self.data = []
        # Eigene Daten
        for path in data_paths:
            with open(path, "rb") as f:
                self.data += pickle.load(f)
        
        # UCI-Daten hinzufügen
        connect_4 = fetch_ucirepo(id=26)
        for features, label in zip(connect_4.data.features.values, connect_4.data.targets.values):
            state = [[' ' if x == 0 else '●' if x == 1 else '○' for x in row] 
                    for row in features.reshape(6,7)]
            move = np.argmax(label)
            self.data.append((state, move))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = Connect4Dataset([
        "connect4_data_3d_10000g.pkl",
        "connect4_data_4d_5000g.pkl"
    ])
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(120):
        model.train()
        total_loss = 0
        for states, moves in loader:
            states, moves = states.to(device), moves.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "connect4_model.pth")


train()