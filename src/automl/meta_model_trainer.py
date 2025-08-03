# import torch
import torch.nn as nn
import pandas as pd

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim * 4, out_features=hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim * 8, out_features=10)
        )
        
    def forward(self, x):
        return self.network(x)
    

df = pd.read_csv("meta-dataset_edited.csv")