from config import Config
from datasets import load_dataset
from model import GCN

import torch

cfg = Config()
device = cfg.device

@torch.no_grad()
def evaluate():
    dataset = load_dataset()

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    X = dataset[0].x.to(device)
    y = dataset[0].y.to(device)
    edge_index = dataset[0].edge_index.to(device)

    test_mask = dataset[0].test_mask

    model = GCN(num_features=num_features, num_hidden=cfg.num_hidden, num_classes=num_classes, dropout=cfg.dropout)
    model = model.to(device)

    model.load_state_dict(torch.load(f'outputs/checkpoints/gcn.pth'))
    model.eval()

    total = test_mask.sum().item()

    outputs = model(X, edge_index)
    _, preds = torch.max(outputs, dim=-1)
    correct = (preds[test_mask] == y[test_mask]).sum().item()

    print(f"Acc: {correct / total * 100: .2f} %") # Accuracy is about 78%

if __name__ == "__main__":
    evaluate()