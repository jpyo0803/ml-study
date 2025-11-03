import torch
import numpy as np
from model import PINN
from config import Config

@torch.no_grad()
def evaluate():
    cfg = Config()
    device = cfg.device
    data = np.load(cfg.data_path)

    # 데이터 로드 및 reshape
    x = data["x"].astype(np.float32)  # (256, 1)
    t = data["t"].astype(np.float32)  # (201, 1)
    u = data["u"].astype(np.float32)  # (256, 201)
    v = data["v"].astype(np.float32)  # (256, 201)
    h = np.sqrt(u**2 + v**2)

    X, T = np.meshgrid(x, t, indexing="ij")  # X:(256,201), T:(256,201)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (256*201, 2)
    u_star = u.flatten()[:, None]
    v_star = v.flatten()[:, None]
    h_star = h.flatten()[:, None]

    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    # 모델 로드
    model = PINN(cfg.layers, lb, ub, device).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/pinn.pth", map_location=device))
    model.eval()

    # 예측 수행
    x_t = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
    t_t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
    u_pred, v_pred = model.forward(x_t, t_t)
    h_pred = torch.sqrt(u_pred**2 + v_pred**2)

    # 에러 계산
    u_star_t = torch.tensor(u_star, dtype=torch.float32, device=device)
    v_star_t = torch.tensor(v_star, dtype=torch.float32, device=device)
    h_star_t = torch.tensor(h_star, dtype=torch.float32, device=device)

    error_u = torch.norm(u_pred - u_star_t, 2) / torch.norm(u_star_t, 2)
    error_v = torch.norm(v_pred - v_star_t, 2) / torch.norm(v_star_t, 2)
    error_h = torch.norm(h_pred - h_star_t, 2) / torch.norm(h_star_t, 2)

    print(f"Eval complete — Error u: {error_u.item():.3e}, v: {error_v.item():.3e}, h: {error_h.item():.3e}")

if __name__ == "__main__":
    evaluate()