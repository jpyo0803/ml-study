from torchvision import datasets, transforms
from config import Config

cfg = Config()

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
])

def load_train_dataset():
    dataset = datasets.MNIST(root=cfg.dataset_path, train=True, download=True, transform=transform)
    return dataset

def load_test_dataset():
    dataset = datasets.MNIST(root=cfg.dataset_path, train=False, download=True, transform=transform)
    return dataset

if __name__ == '__main__':
    load_train_dataset()
    load_test_dataset()
