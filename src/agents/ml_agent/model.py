import torch.nn as nn
import torch.nn.functional as F

class Connect4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        Define the layers of the neural network:
        - conv1: first convolutional layer with 2 input channels, 128 output channels, and a kernel size of 3 with padding 1
        - bn1: first batch normalization layer for 128 channels
        - conv2: second convolutional layer with 128 input channels, 128 output channels, and a kernel size of 3 with padding 1
        - bn2: second batch normalization layer for 128 channels
        - conv3: third convolutional layer with 128 input channels, 64 output channels, and a kernel size of 3 with padding 1
        - bn3: third batch normalization layer for 64 channels
        - dropout: dropout layer with a dropout probability of 0.5
        - fc1: first fully connected layer with input size 64*6*7 and output size 256
        - fc2: second fully connected layer with input size 256 and output size 7
        '''
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        '''
        the method [forward] is used to Define the forward pass of the neural network and pass the data trough the
        diffrent layers.
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x