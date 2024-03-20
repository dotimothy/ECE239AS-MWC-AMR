import torch
import torch.nn as nn
import torch.nn.functional as F

class HisarCNN2019(nn.Module):
    """Implementation of CNN for the Hisar2019 2019.1 Dataset Adopted from RadioML Model (Change Output Layer from 24 to 26 Logits)"""
    def __init__(self):
        super(HisarCNN2019,self).__init__()
        self.name = 'HisarCNN2019'
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 26)
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.selu(self.max_pool(self.conv1(x)))
        x = self.selu(self.max_pool(self.conv2(x)))
        x = self.selu(self.max_pool(self.conv3(x)))
        x = self.selu(self.max_pool(self.conv4(x)))
        x = self.selu(self.max_pool(self.conv5(x)))
        x = self.selu(self.max_pool(self.conv6(x)))
        x = self.selu(self.max_pool(self.conv7(x)))
        x = x.view(-1, 64*8)
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x