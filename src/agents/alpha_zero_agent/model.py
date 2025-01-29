import torch
import torch.nn as nn
import torch.nn.functional as F

class ConnectFourNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Policy Head
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_fc = nn.Linear(32 * rows * cols, cols)
        
        # Value Head
        self.value_conv = nn.Conv2d(128, 3, 1)
        self.value_fc1 = nn.Linear(3 * rows * cols, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value