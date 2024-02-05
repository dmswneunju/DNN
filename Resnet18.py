import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import time
import torchvision
import torchvision.transforms as transforms


# Resnet18 model
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        '''
        in_planes : 입력 필터개수
        out_planes : 출력 필터개수
        '''
        # 3x3 필터를 사용 (너비와 높이를 줄일 때는 stride값 조절)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # 배치정규화
        
        # 3x3 필터를 사용 (패딩1이므로 이미지가 동일하게 나옴)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 핵심 부분
        out = F.relu(out)
        return out
    
    
class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_class=10):
        super(Resnet, self).__init__()
        self.in_planes = 64
        
        # 64개의 3x3필터 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_class)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) # [1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def Resnet18():
    return Resnet(BasicBlock, [2,2,2,2])



# train
def train(epoch):
    print(f"\n[ Train epoch : {epoch}]")
    net.train() # 모델을 학습모드로 설정
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device) # image와 target을 장비에 할당
        optimizer.zero_grad() #optimizer gradient 초기화
              
        outputs = net(inputs) #장비에 할당된 이미지를 모델의 input으로 이용해 output을 계산
        #print("output : ", outputs)
        #print("output.shape : ", outputs.shape) #output.shape :  torch.Size([128, 10]) 128개 이미지 각각 class 10에 대한 확률 
        loss = criterion(outputs, targets) # 계산된 output과 target을 criterion(CrossEntropy)를 이용해서 loss 계산
        #print("loss : ", loss) # loss :  tensor(2.38660)
        loss.backward() # loss 계산한 결과를 바탕으로 back propagation을 통해 계산된 gradient값을 각 파라미터에 할당

        optimizer.step() # gradient값을 이용해 파라미터값 업데이트
        train_loss += loss.item() # tensor에 하나의 값만 존재한다면 scalar값을 얻을 수 있음. 만일 여러개 존재한다면 사용 불가.
              
        '''
        for i in range(outputs.size(1)): #10
            print(outputs[i])'''
        # outputs[0] : 첫번째 이미지에 대한 10개의 클래스 중 확률 값. 첫번째 이미지의 label은 9. 
        # tensor([-0.6967,  0.4949, -0.3854,  0.6380,  0.4872, -0.4960, -1.0212,  0.2237, 0.5431, -0.8949], device='cuda:0', grad_fn=<SelectBackward0>)
        '''        
        for j in range(len(outputs.max(1))):
            print("output max :", outputs.max(1))'''
        _, predicted = outputs.max(1) # output의 크기가 배치크기x클래스개수. 최댓값과 최댓값의 위치를 산출. _으로 처리하여 해당 출력값은 저장하지 않고, 최댓값의 위치만 predicted에 저장하겠다.
        # values=tensor([3.3213, 1.3654, 3.1423, 2.0251, 1.9749, 2.1859, 1.1904, 2.1445, 2.3648, ... ] 128개
        # indices=tensor([5, 9, 1, 1, 5, 8, 9, 5, 7, 5, 0, 8, 7, 1, 9, 5, 4, 1, 1, 5, 1, 5, 6, 2, ... ] => predicted. 128개
        
        total += targets.size(0) # 128
        current_correct = predicted.eq(targets).sum().item() # 배열과 targets가 일치하는지 검사하고 sum으로 일치하는 것들의 개수의 합을 숫자로 출력
        correct += current_correct
        
        if batch_idx % 100 == 0:
            print('\nCurrent batch : ', str(batch_idx))
            print(f'Current batch average train accuracy : {current_correct}/{targets.size(0)} => {current_correct/targets.size(0)}')
            print(f'Current batch average train loss : {loss.item()}/{targets.size(0)} => {loss.item()/targets.size(0)}')
            
    print(f'\nTotal average train accuracy : {correct}/{total} => {correct/total}')
    print(f'Total average train loss : {train_loss}/{total} => {train_loss/total}')
        
        
        
# test
def test(epoch):
    print('\n[Test epoch : %d]' % epoch)
    net.eval()
    loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0) # 128
        
        outputs = net(inputs)
        loss += criterion(outputs, targets).item()
        
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        
    print('\nTotal average test accuracy : ', correct/total)
    print('Total average test loss : ', loss/total)
    
    state = {
        'net' : net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')
    


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 50:
        lr /=10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# Resnet18.py이므로 Resnet18 == __name__
# 직접 실행시켰을 때만 실행되길 원하는 코드들
if __name__ == '__main__':
    # 환경설정
    device = 'cuda'

    net = Resnet18()
    net = net.to(device)
    
    # dataset load
    transform_train = transforms.Compose([ #불러오는 이미지 데이터에 전처리 및 augmentaion을 다양하게 적용할 때 이용하는 메서드
        transforms.RandomCrop(32, padding=4), #잘라낼 크기 설정. 그 크기만큼 랜덤으로 잘라냄
        transforms.RandomHorizontalFlip(), # 해당이미지를 50%의 확률로 좌우반전
        transforms.ToTensor(), #딥러닝 모델의 input으로 이용할 수 있게 이미지 데이터를 tensor형태로 변환 및 0~1로 정규화
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # torchvision dataset의 CIFAR10다운로드
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 모델에 넣기 위한 데이터 세팅
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2) #num_workers는 dataset의 데이터를 gpu로 전송할 때 필요한 전처리를 수행할 때 사용하는 subprocess의 수
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # 손실 함수 및 최적화 알고리즘 정의
    learning_rate = 0.1
    file_name = 'resnet18_cifar10.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002) # weight_decay : 가중치 감소를 통한 가중치 정형화
    
    
    start_time = time.time()

    for epoch in range(0,50):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
        print('\nTime elapsed : ', time.time() - start_time)
