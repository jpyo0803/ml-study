import torch
import torch.nn as nn

class PINN_Burgers(nn.Module):
    def __init__(self, layers, lb, ub, device):
        super().__init__()

        self.iter = 0
 
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

        # Xavier Init 적용
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        u = self.model(H)
        return u
    
    def loss_f(self, X_f_train):
        X_f_train = X_f_train.clone().detach().requires_grad_(True)

        u = self.forward(X_f_train)

        # 전체 입력에 대해 1차 미분
        u_grads = torch.autograd.grad(u, X_f_train, torch.ones_like(u),
                                    retain_graph=True, create_graph=True)[0]
        u_x = u_grads[:, 0:1]
        u_t = u_grads[:, 1:2]

        # 2차 미분 (x에 대해 한 번 더)
        u_xx = torch.autograd.grad(u_x, X_f_train, torch.ones_like(u_x),
                                retain_graph=True, create_graph=True)[0][:, 0:1]

        f = u_t + u * u_x - (0.01 / torch.pi) * u_xx

        loss_f = torch.mean(f**2)
        return loss_f

    def loss_bc(self, X_u_train, u_train):
        u_pred = self.forward(X_u_train)
        loss_bc = torch.mean((u_pred - u_train)**2)
        return loss_bc

    def loss(self, X_f_train, X_u_train, u_train):
        loss_bc = self.loss_bc(X_u_train, u_train)
        loss_f = self.loss_f(X_f_train)
        return loss_bc + loss_f