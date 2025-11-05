import torch
from model import PINN_Burgers
from config import Config  # cfg.device, cfg.layers, cfg.lb, cfg.ub, cfg.data_path 등

@torch.no_grad()
def evaluate():
    cfg = Config()
    device = cfg.device

    # 1. 데이터 로딩
    data = torch.load(cfg.data_path)
    x = data["x"].to(device)     # (256, 1)
    t = data["t"].to(device)     # (201, 1)
    u = data["u"].to(device)     # (256, 201)

    # (x, t)의 전체 조합 생성
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")  # X:(256,201), T:(256,201)
    X = X.flatten()[:, None]  # (256*201, 1)
    T = T.flatten()[:, None]  # (256*201, 1)

    u_star = u.flatten()[:, None]  # (256*201, 1)

    # 2. 모델 로드
    model = PINN_Burgers(cfg.layers, cfg.lb, cfg.ub, device).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/pinn_burgers.pth", map_location=device))
    model.eval()

    # 3. 예측 수행
    X_full = torch.cat([X, T], dim=1).to(device)  # (N, 2)
    u_pred = model.forward(X_full)

    # 4. 상대 L2 오차 계산
    error_u = torch.norm(u_pred - u_star, p=2) / torch.norm(u_star, p=2)

    print(f"Eval complete — Relative L2 Error (u): {error_u.item():.3e}")

if __name__ == "__main__":
    evaluate()
