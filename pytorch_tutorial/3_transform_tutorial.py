import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # PIL Image나 ndarray를 FloatTensor로 변환하고 값을 [0, 1] 사이로 normalize함 
    transform=ToTensor(),
    # 10개의 0을 포함하는 텐서 생성후, scatter_를 통해 첫번째 차원의 y번째 위치에 원하는 값을 넣음
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
