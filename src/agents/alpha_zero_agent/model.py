import torch
import torch.nn as nn
import torch.nn.functional as F

class ConnectFourNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, 1)
        self.policy_fc = nn.Linear(2*rows*cols, cols)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_fc1 = nn.Linear(rows*cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Common trunk
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return F.log_softmax(p, dim=1), v