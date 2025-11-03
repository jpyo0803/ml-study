import torch
from torch.utils.data import Dataset
import numpy as np

# class NLSDataset(Dataset):
#     def __init__(self, x, t, u, v, h):
#         assert len(x) == len(t) and len(t) == len(u) and len(u) == len(v), "Lengths of x, t, u, v must be equal."

#         self.x = torch.tensor(x, dtype=torch.float32) 
#         self.t = torch.tensor(t, dtype=torch.float32)
#         self.u = torch.tensor(u, dtype=torch.float32)
#         self.v = torch.tensor(v, dtype=torch.float32)
#         self.h = torch.tensor(h, dtype=torch.float32)

#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         # sample = {'x': self.x[idx], 't': self.t[idx], 'u': self.u[idx], 'v': self.v[idx]}
#         # return sample
#         return self.x[idx], self.t[idx], self.u[idx], self.v[idx], self.h[idx]

def load_dataset(npz_path, N_0=50, N_b=50, N_f=20000, lb=(-5.0, 0.0), ub=(5.0, np.pi/2)):
    '''
        npz 로드 
    '''

    data = np.load(npz_path)
    x = data['x'].astype(np.float32) # shape: (X, 1)
    t = data['t'].astype(np.float32) # shape: (T, 1)
    u = data['u'].astype(np.float32) # shape: (T, X)
    v = data['v'].astype(np.float32) # shape: (T, X)

    # 초기조건: t = 0에서 (u,v)(x,0)
    N_0 = min(N_0, x.shape[0])
    idx_x = np.random.choice(x.shape[0], N_0, replace=False)
    x0 = x[idx_x, :]                       # (N0, 1)
    u0 = u[idx_x, 0][:, None]              # (N0, 1)
    v0 = v[idx_x, 0][:, None]              # (N0, 1)

    # 경계조건: x = lb, ub 에 대해 다양한 t
    N_b = min(N_b, t.shape[0])
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]                       # (Nb, 1)

    # PDE collocation 포인트
    from pyDOE import lhs
    X_f = np.array(lb) + (np.array(ub) - np.array(lb)) * lhs(2, N_f)

    return x0, u0, v0, tb, X_f, np.array(lb), np.array(ub)

# if __name__ == "__main__":
#     npz_path = 'data/processed/nls_processed.npz'
#     load_dataset(npz_path)