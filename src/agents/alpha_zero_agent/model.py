import torch
import torch.nn as nn
import torch.nn.functional as F

class ConnectFourNet(nn.Module):
    def __init__(self, rows=6, cols=7, action_size=7):
        super(ConnectFourNet, self).__init__()
        self.rows = rows
        self.cols = cols
        self.action_size = action_size
        
        # Gemeinsame Convolutional-Schichten
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Value Head
        self.value_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.value_fc1 = nn.Linear(3 * rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Policy Head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_fc1 = nn.Linear(32 * rows * cols, 64)
        self.policy_fc2 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Value
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 3 * self.rows * self.cols)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        # Policy
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 32 * self.rows * self.cols)
        policy = F.relu(self.policy_fc1(policy))
        policy = self.policy_fc2(policy)
        policy = F.log_softmax(policy, dim=1)
        
        return policy, value