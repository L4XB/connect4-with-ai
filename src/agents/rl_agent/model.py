import torch
import torch.nn as nn

class Connect4DQN(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=256, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Ersetze BatchNorm durch LayerNorm
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Ersetze BatchNorm durch LayerNorm
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Ersetze BatchNorm durch LayerNorm
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def forward(self, x):
        return self.net(x)
    
    def save(self, path='connect4_dqn.pth'):
        torch.save(self.state_dict(), path)
    
    def load(cls, path='connect4_dqn.pth'):
        model = cls()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return model