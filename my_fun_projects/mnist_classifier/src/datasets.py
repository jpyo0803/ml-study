import torch
from torchvision import datasets, transforms
from config import Config

cfg = Config()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_train_dataset():
    train_dataset = datasets.MNIST(root=cfg.dataset_path, train=True, download=True, transform=transform)
    return train_dataset

def load_valid_dataset():
    valid_dataset = datasets.MNIST(root=cfg.dataset_path, train=False, download=True, transform=transform)
    return valid_dataset
