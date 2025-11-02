import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
# 모델 저장
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # 모델 경로를 지정하지 않으면 빈 깡통 모델 반환
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

print(model)

# 모델 가중치만 저장하는 것이 아닌 모델 아키텍처도 같이 저장
torch.save(model, 'model.pth')

model = torch.load('model.pth')