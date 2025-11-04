import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import PINN_NLS
from config import Config
from utils import set_seed, ensure_dirs


def train():
    cfg = Config() # 하이퍼파라미터 및 학습 세팅
    ensure_dirs() # outputs 폴더가 존재하도록 보장
    set_seed(cfg.seed)

    x0, u0, v0, tb, X_f = load_dataset(cfg.data_path, N_0=50, N_b=50, N_f=20000, lb=cfg.lb, ub=cfg.ub)

    x0 = x0.to(cfg.device)
    u0 = u0.to(cfg.device)
    v0 = v0.to(cfg.device)
    tb = tb.to(cfg.device)
    
    x_f = X_f[:, 0:1].to(cfg.device)
    t_f = X_f[:, 1:2].to(cfg.device)

    model = PINN_NLS(cfg.layers, cfg.lb, cfg.ub, cfg.device)
    model = model.to(cfg.device)

    # ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for iter in range(cfg.num_adam_iter):
        optimizer.zero_grad()
        loss = model.loss(x0, u0, v0, tb, x_f, t_f)
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            print(f"Adam iter {iter:05d}, loss={loss.item():.3e}")

    # L-BFGS
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=50000,
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_ = model.loss(x0, u0, v0, tb, x_f, t_f)
        loss_.backward()
        return loss_

    print("Starting L-BFGS optimization...")
    optimizer_lbfgs.step(closure)
    
    '''
        학습 완료 후 모델 가중치 저장
    '''
    torch.save(model.state_dict(), 'outputs/checkpoints/pinn_nls.pth')
    print("Training complete")

if __name__ == "__main__":
    train()




