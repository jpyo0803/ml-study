import torch
import torch.nn as nn

from config import Config
from datasets import load_train_dataset, load_valid_dataset

from torch.utils.data import DataLoader

from model import MNISTClassifier

cfg = Config()

device = cfg.device

def train():
    train_dataset = load_train_dataset()
    valid_dataset = load_valid_dataset()

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = MNISTClassifier(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        model.train()

        train_loss = 0.0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.shape[0]
            train_total += X.shape[0]

        model.eval()
        valid_loss = 0.0
        valid_total = 0

        with torch.no_grad():
            for X, y in valid_loader:
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                valid_loss += loss.item() * X.shape[0]
                valid_total += X.shape[0]

        print(f'Epoch: {epoch + 1} / {cfg.num_epochs}, Train loss: {train_loss / train_total: .4f}, Valid loss: {valid_loss / valid_total: .4f}')

        torch.save(model.state_dict(), 'outputs/checkpoints/mnist_classifier.pth')
        print("Training complete")

if __name__ == '__main__':
    train()