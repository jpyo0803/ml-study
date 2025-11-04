import torch
from torch.utils.data import Dataset

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

def load_dataset(pt_path, N_0=50, N_b=50, N_f=20000, lb=(-5.0, 0.0), ub=(5.0, torch.pi/2)):
    '''
        전처리된 데이터 로딩 (.pt)
    '''

    data = torch.load(pt_path)
    x, t, u, v, h = data['x'], data['t'], data['u'], data['v'], data['h']

    assert N_0 <= x.shape[0], "N_0 must be <= x.shape[0]"
    assert N_b <= t.shape[0], "N_b must be <= t.shape[0]"

    '''
        초기조건 t = 0에서 샘플링된 x값 N_0 랜덤 추출
        t = 0일때 랜덤 위치 x에서의 값   
    '''
    idx_x = torch.randperm(x.shape[0])[:N_0]
    x0 = x[idx_x, :]                       # (N0, 1)
    u0 = u[idx_x, 0][:, torch.newaxis]     # (N0, 1)
    v0 = v[idx_x, 0][:, torch.newaxis]     # (N0, 1)

    '''
        경계조건 x = -5 and 5에서 샘플링된 t값 N_b 랜덤 추출
    '''
    idx_t = torch.randperm(t.shape[0])[:N_b]
    tb = t[idx_t, :]                       # (Nb, 1)

    # PDE collocation 포인트
    '''
        PDE collocation 포인트 추출 
        lb, ub의 왼쪽은 위치, 오른쪽은 시간  
    '''
    from pyDOE import lhs
    lb_t = torch.tensor(lb, dtype=torch.float32)
    ub_t = torch.tensor(ub, dtype=torch.float32)
    X_f = lb_t + (ub_t - lb_t) * lhs(2, N_f)
    X_f = X_f.to(torch.float32)

    return x0, u0, v0, tb, X_f

# if __name__ == "__main__":
#     npz_path = 'data/processed/nls_processed.npz'
#     load_dataset(npz_path)