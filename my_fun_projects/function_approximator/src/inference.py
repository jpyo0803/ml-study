from model import FunctionApproximator
from config import Config
import torch
import matplotlib.pyplot as plt

cfg = Config()

@torch.no_grad()
def inference(x):
    x = x.unsqueeze(1)
    print(x.shape)
    model = FunctionApproximator().to(cfg.device)
    model.load_state_dict(torch.load('outputs/checkpoints/function_approximator.pth'))
    model.eval()

    y = model(x)
    return y

if __name__ == '__main__':
    # Generate inputs
    x = torch.linspace(-5, 5, 100, dtype=torch.float32, device=cfg.device)
    y_true = x**2 + 5 * x - 8

    # Infer
    y_pred = inference(x)

    # Transfer tensors to cpu
    x = x.cpu()
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_true, label='Ground Truth', color='blue')
    plt.plot(x, y_pred, label='Prediction', color='red', linestyle='--')
    plt.title('Function Approximation Result')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()