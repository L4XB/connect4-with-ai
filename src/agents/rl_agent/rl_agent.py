import torch
import torch.nn as nn
import torch.optim as optim

class Connect4DQN(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def save_model(model, path='connect4_dqn.pth'):
    torch.save(model.state_dict(), path)

def load_model(path='connect4_dqn.pth'):
    model = Connect4DQN()
    model.load_state_dict(torch.load(path))
    return model