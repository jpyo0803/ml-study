from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from config import Config

cfg = Config()

transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,)),
])

def load_train_dataset():
    train_dataset = MNIST(root=cfg.dataset_path, train=True, download=True, transform=transforms)
    return train_dataset

def load_test_dataset():
    test_dataset = MNIST(root=cfg.dataset_path, train=False, download=True, transform=transforms)
    return test_dataset
    
if __name__ == '__main__':
    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset()
