import scipy.io
import os
import torch

RAW_PATH = "data/raw/burgers_shock_mu_01_pi.mat"
PROCESSED_DIR = "data/processed/"

def process_burgers_data():
    data = scipy.io.loadmat(RAW_PATH)
    
    t = torch.tensor(data['t'], dtype=torch.float32).flatten().unsqueeze(1) # (100, 1)
    x = torch.tensor(data['x'], dtype=torch.float32).flatten().unsqueeze(1) # (256, 1)

    u = torch.tensor(data['usol'], dtype=torch.float32) # (256, 100)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    torch.save(
        {
            'x': x,
            't': t,
            'u': u,
        },
        os.path.join(PROCESSED_DIR, 'burgers_shock_mu_01_pi_processed.pt')
    )

if __name__ == "__main__":
    process_burgers_data()