import torch
from model import PINN_NLS
from config import Config

@torch.no_grad()
def evaluate():
    cfg = Config()
    device = cfg.device

    '''
        데이터 로딩
        x, t의 모든 조합을 생성 (카테시안 그리드)

        x: (256, 1)
        t: (201, 1)
        u, v, h: (256, 201)
    '''
    data = torch.load(cfg.data_path)

    x, t, u, v, h = data['x'], data['t'], data['u'], data['v'], data['h']

    x = x.to(cfg.device)
    t = t.to(cfg.device)
    u = u.to(cfg.device)
    v = v.to(cfg.device)
    h = h.to(cfg.device)

    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")  # X:(256,201), T:(256,201)
    X = X.flatten()[:, torch.newaxis]
    T = T.flatten()[:, torch.newaxis]
    
    u_star = u.flatten()[:, torch.newaxis] # (256*201, 1)
    v_star = v.flatten()[:, torch.newaxis] # (256*201, 1)
    h_star = h.flatten()[:, torch.newaxis] # (256*201, 1)

    # 모델 로드
    model = PINN_NLS(cfg.layers, cfg.lb, cfg.ub, device).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/pinn_nls.pth", map_location=device))
    model.eval()

    # 예측 수행
    u_pred, v_pred = model.forward(X, T)
    h_pred = torch.sqrt(u_pred**2 + v_pred**2)

    '''
        상대 L2 오차 측정
        Error = ||u_pred-u_true||_2 / ||u_true||_2
    '''
    error_u = torch.norm(u_pred - u_star, p=2) / torch.norm(u_star, p=2)
    error_v = torch.norm(v_pred - v_star, p=2) / torch.norm(v_star, p=2)
    error_h = torch.norm(h_pred - h_star, p=2) / torch.norm(h_star, p=2)

    print(f"Eval complete — Error u: {error_u.item():.3e}, v: {error_v.item():.3e}, h: {error_h.item():.3e}")

if __name__ == "__main__":
    evaluate()