import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu" 
        )
        self.data_path = "data/processed/burgers_shock_mu_01_pi_processed.pt"
        self.layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.N_u = 100
        self.N_f = 10_000
        self.lb = [-1.0, 0.0]
        self.ub = [1.0, 1.0]
        self.seed = 1234
