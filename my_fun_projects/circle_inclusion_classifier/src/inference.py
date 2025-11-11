import torch
import torch.nn as nn

from model import CircleInclusionClassifier
from config import Config

cfg = Config()

@torch.no_grad()
def predict(X):
    X = X.to(cfg.device)

    model = CircleInclusionClassifier().to(cfg.device)
    model.load_state_dict(torch.load('outputs/checkpoints/circle_inclusion_classifier.pth'))
    model.eval()

    y = model(X)
    return y

if __name__ == '__main__':
    X = torch.rand((100, 2)) * 2 - 1
    y = ((X[:, 0:1]**2) + (X[:, 1:2]**2) < 1.0).float()
    pred = predict(X)
    pred = pred > 0.5
    pred = pred.cpu()

    total = X.shape[0]
    correct = (y == pred).sum().item()
    print(f'Accuracy: {correct / total * 100: .2f} %') # 97% accurate