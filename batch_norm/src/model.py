import torch
import torch.nn as nn

from batch_norm_1d import MyBatchNorm1D

class MNISTClassifier(nn.Module):
    def __init__(self, enable_bn=False):
        super().__init__()

        layers = []

        prev = 28 * 28
        for _ in range(3):
            layers.append(nn.Linear(prev, 100, bias=not enable_bn))
            if enable_bn:
                layers.append(MyBatchNorm1D(100))
            layers.append(nn.Sigmoid())
            prev = 100
        layers.append(nn.Linear(100, 10, bias=not enable_bn))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)
