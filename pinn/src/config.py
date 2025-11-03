import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.data_path = "data/processed/nls_processed.npz"
        self.lb = [-5.0, 0.0]                # domain lower bound (x_min, t_min)
        self.ub = [5.0, 1.57]                # domain upper bound (x_max, t_max)
        self.layers = [2, 100, 100, 100, 100, 2]  # network layer sizes
        self.batch_size = 256
        self.lr = 1e-3
        self.epochs = 2000
        self.n_samples = 10000
        self.seed = 42

# if __name__ == "__main__":
#     config = Config()