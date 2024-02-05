from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# from 파일 import 함수
from Resnet18 import Resnet18 



# 새로운 이미지 로드
image = Image.open('./ship.jpg')
# 모델에 맞게 사이즈 변형
reshape_image = image.resize((32,32))

'''fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title('image')
print(image.size)

plt.subplot(1,2,2)
plt.imshow(reshape_image)
plt.title('reshape image')
print(reshape_image.size)'''

# tensor로 올리기 위한 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

cat_tensor = transform(reshape_image).unsqueeze(0)

# gpu로 보내기 전에 모델 생성
net = Resnet18()

# gpu 준비단계
if torch.cuda.is_available():
    print(f'torch is available : {torch.cuda.is_available()}')
    net = net.to('cuda')
    
if torch.cuda.is_available():
    cat_tensor = cat_tensor.to('cuda')
    
# pretrain 준비
pretrained_dict = torch.load('./checkpoint/resnet18_cifar10.pth')
net_dict = net.state_dict()

# 필요한 가중치만 가져오기
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}

# 선택한 가중치만 불러오기
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)

with torch.no_grad():  # 기울기 계산 비활성화
    output = net(cat_tensor)
    predicted_class = output.argmax(1).item()
    # torch.return_types.max(
    # values=tensor([4.0649], device='cuda:0'),
    # indices=tensor([3], device='cuda:0'))
    
# 예측을 확인
print("예측된 클래스:", predicted_class)
