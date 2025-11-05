import torch

'''
    전체 데이터셋 리턴 
'''
def load_dataset(pt_path: str):
    data = torch.load(pt_path)

    x = data['x']
    t = data['t']

    dataset = {
        'x': data['x'],
        't': data['t'],
        'u': data['u'],
    }
    return dataset

'''
    전체 데이터셋에서 학습 데이터를 랜덤 샘플링 후 리턴 
'''
def generate_train_dataset(dataset: dict, N_u=100, N_f=10_000,
                            lb=(-1.0, 0.0), ub=(1.0, 1.0)):
    x, t, u = dataset['x'], dataset['t'], dataset['u']

    X, T = torch.meshgrid(x.flatten(), t.flatten(), indexing='ij')

    # 초기 조건: t = 0
    leftedge_x = torch.stack([X[:, 0], T[:, 0]], dim=-1) # (256, 2)
    leftedge_u = u[:, 0:1] # (256, 1)

    # 경계 조건 x = -1
    topedge_x = torch.stack([X[0, :], T[0, :]], dim=-1) # (100, 2)
    topedge_u = u[0:1, :].T # (100, 1)

    # 경계 조건 x = 1
    bottomedge_x = torch.stack([X[-1, :], T[-1, :]], dim=-1) # (100, 2)
    bottomedge_u = u[-2:-1, :].T # (100, 1)

    all_X_u_train = torch.cat([leftedge_x, topedge_x, bottomedge_x], dim=0) # (456, 2)
    all_u_train = torch.cat([leftedge_u, topedge_u, bottomedge_u], dim=0) # (456, 1)

    idx = torch.randperm(all_X_u_train.shape[0])[:N_u]

    X_u_train = all_X_u_train[idx, :]
    u_train = all_u_train[idx, :]

    # Collocation point 생성
    from pyDOE import lhs
    lb_t = torch.tensor(lb, dtype=torch.float32)
    ub_t = torch.tensor(ub, dtype=torch.float32)
    X_f_train = lb_t + (ub_t - lb_t) * lhs(2, N_f)
    X_f_train = torch.cat([X_f_train, X_u_train], dim=0)
    X_f_train = X_f_train.to(torch.float32)

    return X_f_train, X_u_train, u_train

if __name__ == "__main__":
    dataset = load_dataset('data/processed/burgers_shock_mu_01_pi_processed.pt')
    generate_train_dataset(dataset)