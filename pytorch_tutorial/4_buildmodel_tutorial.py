import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''
    만약 cuda가 존재한다면 cuda 사용
    만약 mps가 존재한다면 mps 사용
    아니면 그냥 cpu 사용 
'''
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

'''
    nn.Module을 상속하면 모듈 내부의 모든 필드들이 자동 추적되며 
    parameters() 및 named_parameters() 메서드로 접근 가능 
'''
class NeuralNetwork(nn.Module):
  def __init__(self):
    '''
        신경망 계층을 초기화 
    '''
    super().__init__() # nn.Module을 상속하면 항상 맨 처음 호출해주기 

    # flatten의 첫 입력은 batch 개수로 약속 
    self.flatten = nn.Flatten()

    # Sequential은 순서를 갖는 모듈의 컨테이너임, 나열된 순서대로 입력이 각 모듈을 거쳐 다음 모듈로 파이프라인 따라 전달 
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(in_features=28 * 28, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=10),
    )

  def forward(self, x):
    '''
        nn.Module을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산들을 구현 
    '''
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
  
'''
    모델을 생성 후, 타겟 디바이스로 모델을 이동하고 모델 구조 출력 
'''
model = NeuralNetwork().to(device)
print(model)

for name, param in model.named_parameters():
  print(f'Param name: {name}, size: {param.size()}, values: {param[:2]}')

X = torch.rand(2, 28, 28, device=device)

# 입력 X를 모델에 넣음. 절대 forward()를 직접 부르지 말것 (백그라운드 추가작업이 실행되지않음)! 
logits = model(X)
pred_prob = nn.Softmax(dim=1)(logits) # dim은 값의 합이 1이되어야하는 차원 
y_pred = pred_prob.argmax(dim=1) # dim은 최대값의 위치를 찾을때의 관심 차원 
print("y_pred: ", y_pred)
