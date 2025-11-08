import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.batch_size = 128
        self.lr = 0.01
        self.num_epochs = 100
