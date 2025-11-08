import torch
import torch.nn as nn

from datasets import load_train_dataset
from model import FunctionApproximator
from config import Config

cfg = Config()

def train():
    train_dataset = load_train_dataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    model = FunctionApproximator()
    model = model.to(cfg.device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.num_epochs):
        model.train()
        
        running_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            x = x.unsqueeze(dim=1)
            y = y.unsqueeze(dim=1)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.sum().item()
            total += x.shape[0]
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch} Loss: {running_loss / total:.4f}')

    torch.save(model.state_dict(), 'outputs/checkpoints/function_approximator.pth')
    print("Training complete")


if __name__ == '__main__':
    train()