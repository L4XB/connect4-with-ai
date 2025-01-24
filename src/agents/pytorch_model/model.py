import torch
import torch.nn as nn

from constants import  AMOUNT_COLUMNS, AMOUNT_ROWS

class PytorchConnectFourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(AMOUNT_COLUMNS * AMOUNT_ROWS, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)
        
    def forward(self, x):
        x = x.view(-1, AMOUNT_COLUMNS * AMOUNT_ROWS)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x