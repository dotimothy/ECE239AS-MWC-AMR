import torch
import torch.nn as nn
import torch.nn.functional as F

class OSheaCNN2016(nn.Module):
    """Implementation of CNN for RadioML 2016.10A """
    def __init__(self):
        super(OSheaCNN2016,self).__init__()
        self.name = 'OSheaCNN2016'
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*128, 128)
        self.output = nn.Linear(128, 11)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x.view(-1, 16*128)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.output(x)
        x = self.softmax(x)
        return x