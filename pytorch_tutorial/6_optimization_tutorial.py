import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

learning_rate = 1e-3
batch_size = 64 # 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수 
epochs = 5 # 데이터셋을 반복하는 횟수

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    # 모델을 train 모드로 설정. train모드에서만 batch normalization과 dropout 적용
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward() # loss를 역전파
        optimizer.step() # 모델 가중치 업데이트 
        optimizer.zero_grad() # (다음을 위해) gradient 누적을 방지하기 위해 0으로 초기화.

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)

    # 모델을 evavl 모드로 설정
    model.eval()

    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = NeuralNetwork().to(device)
print(model)



# 손실 함수를 초기화 
loss_fn = nn.CrossEntropyLoss()

'''
    여기서는 SGD 사용함. 
    PyTorch에서는 ADAM이나 RMSProp과 같은 다른 종류의 모델과 데이터에서
    더 잘 동작하는 다양한 옵티마이저를 지원 
'''
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")