import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from src.agents.ml_agent.model import Connect4MLP

class Connect4Dataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, move = self.data[idx]
        return torch.FloatTensor(state), torch.LongTensor([move])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = Connect4Dataset("connect4_data_3d_5000g.pkl")
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