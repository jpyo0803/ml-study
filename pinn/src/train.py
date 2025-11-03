import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import PINN
from config import Config
from utils import set_seed, ensure_dirs
import numpy as np


def main():
    cfg = Config() # 하이퍼파라미터 및 학습 세팅
    ensure_dirs() # outputs 폴더가 존재하도록 보장
    set_seed(cfg.seed)

    x0, u0, v0, tb, X_f, lb, ub = load_dataset(cfg.data_path, N_0=50, N_b=50, N_f=20000, lb=(-5.0, 0.0), ub=(5.0, np.pi/2))

    x0 = torch.tensor(x0, dtype=torch.float32, device=cfg.device)
    u0 = torch.tensor(u0, dtype=torch.float32, device=cfg.device)
    v0 = torch.tensor(v0, dtype=torch.float32, device=cfg.device)
    tb = torch.tensor(tb, dtype=torch.float32, device=cfg.device)
    x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=cfg.device)
    t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=cfg.device)

    model = PINN(cfg.layers, cfg.lb, cfg.ub, cfg.device)
    model = model.to(cfg.device)

    # ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for it in range(2000):
        optimizer.zero_grad()
        loss = model.loss(x0, u0, v0, tb, x_f, t_f)
        loss.backward()
        optimizer.step()
        if it % 100 == 0:
            print(f"Adam iter {it:05d}, loss={loss.item():.3e}")

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
    
    torch.save(model.state_dict(), 'outputs/checkpoints/pinn.pth')
    print("Training complete")

if __name__ == "__main__":
    main()




