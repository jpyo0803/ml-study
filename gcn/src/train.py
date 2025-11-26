import torch
import torch.nn as nn

from config import Config
from datasets import load_dataset
from model import GCN
from torch.utils.data import DataLoader

cfg = Config()

device = cfg.device

def train():
    dataset = load_dataset()

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    X = dataset[0].x.to(device)
    y = dataset[0].y.to(device)
    edge_index = dataset[0].edge_index.to(device)

    train_mask = dataset[0].train_mask
    valid_mask = dataset[0].val_mask

    model = GCN(num_features=num_features, num_hidden=cfg.num_hidden, num_classes=num_classes, dropout=cfg.dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.NLLLoss()

    min_valid_loss = 1e9
    early_stop_cnt = 0

    for epoch in range(cfg.num_epochs):
        model.train()

        train_loss = 0.0

        optimizer.zero_grad()
        outputs = model(X, edge_index)
        loss = criterion(outputs[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        model.eval()

        valid_loss = 0.0

        with torch.no_grad():
            outputs = model(X, edge_index)
            loss = criterion(outputs[valid_mask], y[valid_mask])

            valid_loss = loss.item()
            
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
            else:
                early_stop_cnt += 1
                if early_stop_cnt > cfg.early_stopping:
                    print("Early stopping")
                    break

        print(f'Epoch: {epoch + 1} / {cfg.num_epochs}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}')

    torch.save(model.state_dict(), f'outputs/checkpoints/gcn.pth')
    print("Training complete")

if __name__ == '__main__':
    train()