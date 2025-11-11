import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

def load_train_dataset():
    X = torch.rand((10000, 2)) * 2 - 1
    y = ((X[:, 0:1]**2) + (X[:, 1:2]**2) < 1.0).float()
    dataset = CustomDataset(X, y)
    return dataset

def load_test_dataset():
    X = torch.rand((2000, 2)) * 2 - 1
    y = ((X[:, 0:1]**2) + (X[:, 1:2]**2) < 1.0).float()
    dataset = CustomDataset(X, y)
    return dataset

if __name__ == '__main__':
    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset()