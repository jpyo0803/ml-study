from datasets import load_test_dataset
import torch
from model import FunctionApproximator
from config import Config

cfg = Config()

@torch.no_grad()
def evaluate():
    test_dataset = load_test_dataset()

    test_loader = torch.utils.data.DataLoader(test_dataset)

    model = FunctionApproximator().to(cfg.device)
    model.load_state_dict(torch.load('outputs/checkpoints/function_approximator.pth'))
    model.eval()

    running_loss = 0.0
    total = 0

    for x, y in test_loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        outputs = model(x)

        running_loss += torch.mean((outputs - y)**2).sum().item()
        total += x.shape[0]
    
    print("Loss: ", running_loss / total)


if __name__ == '__main__':
    evaluate()