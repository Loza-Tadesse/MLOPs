"""
Improved DNN Model for Price Prediction
Replaces LSTM with a simpler but more effective architecture
"""

import torch
import torch.nn as nn


class ImprovedDNN(nn.Module):
    """Improved Deep Neural Network for Price Prediction"""
    
    def __init__(self, input_size, dropout_rate=0.3):
        super(ImprovedDNN, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 1
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Output layer
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)
