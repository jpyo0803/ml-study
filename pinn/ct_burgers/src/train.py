import torch 

from model import PINN_Burgers
from datasets import load_dataset, generate_train_dataset
from config import Config
from utils import set_seed, ensure_dirs

def train():
    cfg = Config()
    ensure_dirs()
    set_seed(cfg.seed)

    dataset = load_dataset(cfg.data_path)
    X_f_train, X_u_train, u_train = generate_train_dataset(dataset, N_u=cfg.N_u, N_f=cfg.N_f, lb=cfg.lb, ub=cfg.ub)

    X_f_train = X_f_train.to(cfg.device)
    X_u_train = X_u_train.to(cfg.device)
    u_train = u_train.to(cfg.device)

    model = PINN_Burgers(cfg.layers, cfg.lb, cfg.ub, cfg.device)
    model = model.to(cfg.device)

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(20000):
        optimizer_adam.zero_grad()
        loss = model.loss(X_f_train, X_u_train, u_train)
        loss.backward()
        optimizer_adam.step()
        if i % 1000 == 0:
            print(f"[Adam] iter {i}, loss = {loss.item():.4e}")

    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, 
                              max_iter = 250, 
                              max_eval = None, 
                              tolerance_grad = 1e-05, 
                              tolerance_change = 1e-09, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = model.loss(X_f_train, X_u_train, u_train)
        loss.backward()   
        # model.iter += 1
        # if model.iter % 100 == 0:
        #     error_vec, _ = PINN.test()
        #     print(loss,error_vec)
        return loss 
 
    print("Starting L-BFGS optimization...")
    optimizer_lbfgs.step(closure)
    
    '''
        학습 완료 후 모델 가중치 저장
    '''
    torch.save(model.state_dict(), 'outputs/checkpoints/pinn_burgers.pth')
    print("Training complete")

if __name__ == "__main__":
    train()