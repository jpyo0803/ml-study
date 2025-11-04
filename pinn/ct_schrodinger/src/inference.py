import torch
from model import PINN_NLS
from config import Config

def inference(x_query, t_query):
    """
    x_query, t_query: torch tensor
    예: x_query = torch.linspace(-5, 5, 100)[:,None]
        t_query = torch.full_like(x_query, torch.pi/4)
    """

    cfg = Config()
    device = cfg.device

    x_query = x_query.to(device)
    t_query = t_query.to(device)

    # 모델 로드
    model = PINN_NLS(cfg.layers, cfg.lb, cfg.ub, device).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/pinn_nls.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        u_pred, v_pred = model.forward(x_query, t_query)
        h_pred = torch.sqrt(u_pred**2 + v_pred**2)

    return (
        u_pred.cpu(),
        v_pred.cpu(),
        h_pred.cpu(),
    )

if __name__ == "__main__":
    xq = torch.linspace(-5, 5, 100)[:, torch.newaxis]
    tq = torch.full_like(xq, torch.pi / 4)
    u, v, h = inference(xq, tq)
    print("Inference done. u, v, h:", u, v, h)