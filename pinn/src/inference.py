import torch
import numpy as np
from model import PINN
from config import Config

def inference(x_query, t_query):
    """
    x_query, t_query: numpy array or list
    예: x_query = np.linspace(-5, 5, 100)[:,None]
        t_query = np.full_like(x_query, np.pi/4)
    """

    cfg = Config()
    device = cfg.device
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    # 모델 로드
    model = PINN(cfg.layers, lb, ub, device).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/pinn.pth", map_location=device))
    model.eval()
    # 입력 준비
    x_t = torch.tensor(x_query, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_query, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred, v_pred = model.forward(x_t, t_t)
        h_pred = torch.sqrt(u_pred**2 + v_pred**2)

    return (
        u_pred.cpu().numpy(),
        v_pred.cpu().numpy(),
        h_pred.cpu().numpy(),
    )

if __name__ == "__main__":
    xq = np.linspace(-5, 5, 100)[:, None]
    tq = np.full_like(xq, np.pi / 4)
    u, v, h = inference(xq, tq)
    print("Inference done. u, v, h:", u, v, h)