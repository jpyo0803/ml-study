import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers, lb, ub, device):
        super().__init__()

        # 스케일링 경계는 텐서로 들고 있어야 forward에서 브로드캐스트 쉬움
        self.device = device
        self.lb = torch.tensor(lb, dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=self.device)

        net = []
        # layers는 각 layer의 크기를 갖음, 마지막 layer는 마지막에 이어 붙힘 
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))

        # List의 원소들을 unpacking하고 개별 요소들을 파라미터로 넘김 
        self.model = nn.Sequential(*net)


    def forward(self, x, t):
        X = torch.cat([x, t], dim=1) # shape: (B, 2)
        # 입력 정규화: H = [-1, 1], 수렴/일반화에 도움
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.model(H) # shape: (B, 2)
        return uv[:, 0:1], uv[:, 1:2]

    def net_uv(self, x, t):
        u, v = self.forward(x, t)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        u, v, u_x, v_x = self.net_uv(x, t)

        u_t = torch.autograd.grad(
            u,                  # 미분 대상
            t,                  # 이 변수로 미분
            torch.ones_like(u), # 기울기 seed (벡터-자코비안 product)# 
            retain_graph=True, 
            create_graph=True   # 결과물인 u_t를 다시 미분 가능하게함. 아마도 t에대해서는 필요 없을듯 하나 남겨둠 
        )[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]

        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v


    def loss(self, x0, u0, v0, tb, x_f, t_f):
        # 초기 조건
        t0 = torch.zeros_like(x0).to(self.device)
        u0_pred, v0_pred = self.forward(x0, t0)
        loss_ic = torch.mean((u0_pred - u0)**2) + torch.mean((v0_pred - v0)**2)

        # 경계 조건
        x_lb = torch.full_like(tb, self.lb[0], requires_grad=True).to(self.device)
        x_ub = torch.full_like(tb, self.ub[0], requires_grad=True).to(self.device)

        u_lb, v_lb, u_x_lb, v_x_lb = self.net_uv(x_lb, tb)
        u_ub, v_ub, u_x_ub, v_x_ub = self.net_uv(x_ub, tb)

        loss_bc = (
            torch.mean((u_lb - u_ub)**2)
            + torch.mean((v_lb - v_ub)**2)
            + torch.mean((u_x_lb - u_x_ub)**2)
            + torch.mean((v_x_lb - v_x_ub)**2)
        )

        f_u, f_v = self.net_f_uv(x_f, t_f)
        loss_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        return loss_ic + loss_bc + loss_pde
