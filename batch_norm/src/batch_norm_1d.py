import torch
import torch.nn as nn

class MyBatchNorm1D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))

        # 학습되지않는 파라미터 
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.ones(1, num_features))

        self.eps = eps
        self.momentum = momentum

    def forward(self, X):
        if self.training:
            batch_mean = torch.mean(X, dim=0, keepdim=True)
            batch_var = torch.var(X, dim=0, keepdim=True, unbiased=False) # 모집단 분산 사용

            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            
            X_hat = (X - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            X_hat = (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * X_hat + self.beta

