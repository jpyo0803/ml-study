import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.dataset_path = './data/processed/'

        self.num_epochs = 5
        self.lr = 1e-3
        self.batch_size = 64