import torch
import torch.nn as nn

from datasets import load_train_dataset, load_test_dataset
from torch.utils.data import DataLoader
from config import Config
from model import MNISTClassifier

cfg = Config()

device = cfg.device

def train():
    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset()

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model_base = MNISTClassifier().to(device) # baseline
    model_bn = MNISTClassifier(enable_bn=True).to(device) # baseline

    criterion_base = nn.CrossEntropyLoss()
    criterion_bn = nn.CrossEntropyLoss()

    optimizer_base = torch.optim.SGD(model_base.parameters(), lr=cfg.base_lr) # no momentum
    optimizer_bn = torch.optim.SGD(model_bn.parameters(), lr=cfg.base_lr) # no momentum

    pairs_train = [(model_base, criterion_base, optimizer_base), (model_bn, criterion_bn, optimizer_bn)]
    pairs_eval = [(model_base, criterion_base), (model_bn, criterion_bn)]

    base_train_avg_losses = []
    base_val_avg_losses = []

    bn_train_avg_losses = []
    bn_val_avg_losses = []

    for epoch in range(cfg.num_epochs):
        model_base.train()
        model_bn.train()

        base_train_loss = 0.0
        base_train_total = 0
        
        bn_train_loss = 0.0
        bn_train_total = 0

        for X, y in train_loader:
            X = X.to(device).flatten(1, -1)
            y = y.to(device)

            for i, (model, criterion, optimizer) in enumerate(pairs_train):
                optimizer.zero_grad()

                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()

                optimizer.step()

                if i == 0:
                    base_train_loss += loss.item() * X.shape[0]
                    base_train_total += X.shape[0]
                else:
                    bn_train_loss += loss.item() * X.shape[0]
                    bn_train_total += X.shape[0]

        model_base.eval()
        model_bn.eval()

        base_val_loss = 0.0
        base_val_total = 0
        
        bn_val_loss = 0.0
        bn_val_total = 0

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device).flatten(1, -1)
                y = y.to(device)

                for i, (model, criterion) in enumerate(pairs_eval):

                    outputs = model(X)
                    loss = criterion(outputs, y)

                    if i == 0:
                        base_val_loss += loss.item() * X.shape[0]
                        base_val_total += X.shape[0]
                    else:
                        bn_val_loss += loss.item() * X.shape[0]
                        bn_val_total += X.shape[0]

        print(f'Epoch: {epoch + 1} / {cfg.num_epochs}')
        print(f'\t[Base] Train loss: {base_train_loss / base_train_total: .4f}, Val loss: {base_val_loss / base_val_total: .4f}')
        print(f'\t[Batch Norm] Train loss: {bn_train_loss / bn_train_total: .4f}, Val loss: {bn_val_loss / bn_val_total: .4f}')

        base_train_avg_losses.append(base_train_loss / base_train_total)
        base_val_avg_losses.append(base_val_loss / base_val_total)
        
        bn_train_avg_losses.append(bn_train_loss / bn_train_total)
        bn_val_avg_losses.append(bn_val_loss / bn_val_total)

    return base_train_avg_losses, base_val_avg_losses, bn_train_avg_losses, bn_val_avg_losses

import matplotlib.pyplot as plt

def plot_losses(base_train, base_val, bn_train, bn_val):
    epochs = list(range(1, len(base_train) + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, base_train, label='Base Train Loss', linewidth=2)
    plt.plot(epochs, base_val, label='Base Val Loss', linewidth=2)
    plt.plot(epochs, bn_train, label='BN Train Loss', linewidth=2)
    plt.plot(epochs, bn_val, label='BN Val Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves: Base vs BatchNorm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    base_train_avg_losses, base_val_avg_losses, bn_train_avg_losses, bn_val_avg_losses = train()
    plot_losses(base_train_avg_losses, base_val_avg_losses, bn_train_avg_losses, bn_val_avg_losses)