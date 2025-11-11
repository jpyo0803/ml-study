import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from model import CircleInclusionClassifier
from datasets import load_train_dataset, load_test_dataset
from config import Config

cfg = Config()

device = cfg.device

def train():
    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset()

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    model = CircleInclusionClassifier().to(device)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(cfg.num_epochs):
        model.train()

        train_loss = 0
        train_total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * X.shape[0]
            train_total += X.shape[0]
        
        model.eval()

        valid_loss = 0
        valid_total = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                valid_loss += loss.item() * X.shape[0]
                valid_total += X.shape[0]
        
        scheduler.step()

        print(f'[Epoch: {epoch} / {cfg.num_epochs}], Train Loss: {train_loss / train_total: .4f}, Valid Loss: {valid_loss / valid_total: .4f}')

    torch.save(model.state_dict(), 'outputs/checkpoints/circle_inclusion_classifier.pth')
    print("Training complete")

if __name__ == '__main__':
    train()