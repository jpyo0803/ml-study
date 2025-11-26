import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.lr = 0.01
        self.num_epochs = 200
        self.num_hidden = 16
        self.weight_decay = 5e-4
        self.early_stopping = 10
        self.max_degree = 3
        self.dropout = 0.5