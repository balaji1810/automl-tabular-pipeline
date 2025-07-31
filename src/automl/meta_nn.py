import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MetaNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.net(x)
