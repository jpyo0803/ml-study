import torch
import torch.nn as nn

class FunctionApproximator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 128)
        self.tanh1 = nn.Tanh() # smooth 비선형 
        self.fc2 = nn.Linear(128, 128)
        self.tanh2 = nn.Tanh() # smooth 비선형 
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        return out