import torch
from torch.utils.data import TensorDataset

def load_train_dataset():
    x = torch.linspace(-5, 5, 12800)
    y = x**2 + 5 * x - 8

    dataset = TensorDataset(x, y)
    return dataset

def load_test_dataset():
    x = torch.linspace(-5, 5, 12800)
    y = x**2 + 5 * x - 8

    dataset = TensorDataset(x, y)
    return dataset

if __name__ == '__main__':
    dataset = load_train_dataset()
    print(dataset)