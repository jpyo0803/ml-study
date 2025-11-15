import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.dataset_path = './data/processed/'
        self.batch_size = 60
        self.num_epochs = 50 # (# iters) / ((# of MNIST image) / batch_size)

        self.base_lr = 0.1