import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data", # 데이터 저장 경로 
    train=True, # 학습용 데이터 여부 
    download=True, # 만약 데이터가 존재하지 않으면 인터넷에서 다운로드 
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label]) # 가장 최근 생성된 subplot에 title 추가 
    plt.axis("off") # 눈금 제거 
    plt.imshow(img.squeeze(), cmap="gray") # img는 [1, H, W]인데 imshow는 [H, W]을 받음 
plt.show()

import os # 파일 경로를 합치거나 확인 
import pandas as pd # csv 파일을 읽어서 이미지 파일 이름과 라벨을 관리 
from torchvision.io import read_image # 이미지를 tensor 형섹으로 읽음 [C, H, W]

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        '''
            annotation_file: 이미지 파일 이름과 라벨이 적힌 csv 파일 경로
            img_dir: 실제 이미지 파일이 들어있는 폴더 경로
            transform: 이미지에 적용할 변환 (Resize, Normalize, ToTensor 등)
            target_transform: 라벨에 적용할 변환 
        '''

        # csv를 읽어서 dataframe 형태로 저장 
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        # 이미지 폴더 경로 저장 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # 데이터셋의 전체 크기를 반환(이미지 개수)
        return len(self.img_labels)

    '''
        인덱스를 하나 받아서 해당하는 이미지와 라벨을 반환하는 함수 
    '''
    def __getitem__(self, idx):
        # 이미지 저장 위치와 이미지 파일명을 합침, pandas에서는 iloc으로 행렬 접근 
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # Dict 자료형에 넣어서 반환, 필요시 다른 key-value 쌍도 추가 가능 
        sample = {"image": image, "label": label}
        return sample

from torch.utils.data import DataLoader

'''
    DataSet은 데이터셋의 특징을 가져오고 하나의 샘플에 라벨을 지정하는 일을 한번에 함.
    모델을 학습할때 샘플들을 미니배치로 전달하고 매 에포크마다 데이터를 섞어 과적합을 방지함.

    DataLoader는 간단한 API로 이러한 복잡한 과정을 추상화한 순회 가능한 객체임 
'''
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# dataloader의 iterator를 반환하고 바로 다음 배치를 가져옴 (맨 처음이니 첫번째 )
train_features, train_labels = next(iter(train_dataloader))
print("Train features shape: ", train_features.size()) # [64, 1, 28, 28]
print("Train labels shape: ", train_labels.size()) # [64]
img = train_features[0].squeeze() # 첫번째 샘플 이미지를 가져와 1인 차원을 제거, 결과 shape[28, 28]
print("Image shape after squeeze: ", img.size())
label = train_labels[0]
plt.imshow(img, cmap='gray')
plt.show()
print("Label: ", label)

