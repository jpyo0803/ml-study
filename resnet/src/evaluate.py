from config import Config
from datasets import load_test_dataset
from model import ResNet_CIFAR 

import torch

cfg = Config()

@torch.no_grad()
def evaluate():
    test_dataset = load_test_dataset()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    device = cfg.device

    model = ResNet_CIFAR(depth=cfg.depth)
    model = model.to(device)
    model.load_state_dict(torch.load(f'outputs/checkpoints/resnet_{cfg.depth}.pth'))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        _, predicted = outputs.max(dim=1)

        total += inputs.shape[0]
        correct += (predicted == targets).sum().item()

    print(f"Acc: {correct / total * 100: .2f} %") # Accuracy is about 91%

if __name__ == "__main__":
    evaluate()