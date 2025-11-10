import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.ps.is_available()
            else 'cpu'
        )
        self.dataset_path = 'data/processed/'
        self.latent_dim = 100        
        self.hidden_dim = 256
        self.img_dim = 28 * 28
        self.batch_size = 128
        self.lr = 0.0001
        self.num_epochs = 50