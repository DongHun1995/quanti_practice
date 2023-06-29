import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from dongnet import dongnet12

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

def quantize(X, NBIT):
    # 1. find threshold
    alpha = np.max(X)
    beta = np.min(X)
    alpha_q = -2**(NBIT - 1)
    beta_q = 2**(NBIT - 1) - 1

    s = (beta - alpha) / (beta_q - alpha_q)
    z = int((beta*alpha_q - alpha * beta_q) / (beta - alpha))

    data_q = np.round(1/s * X + z, decimals=0)
    data_q = np.clip(data_q, alpha_q, beta_q)    
    data_q = data_q.astype(np.int8)
        
    data_qn = data_q
    data_qn = data_qn.astype(np.int32)
    data_qn = s * (data_qn - z)
    data_qn = data_qn.astype(np.float32)
    
    return data_qn

def evaluate_model(model, test_loader, device):
    #batch norm 고정, dropout 안함, gradient 계산안함
    model.eval()
    #model 파라미터를 지정한 device 메모리에 올림
    model.to(device)

    running_corrects = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        #텐서의 최대값과 index를 반환
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # inputs.size(0) 현재 mini-batch의 크기(input의 0번째 dimension 크기)
        running_corrects += torch.sum(preds == labels.data)
    
    eval_accuracy = running_corrects / len(test_loader.dataset)

    print("Dongnet12 cifar100 Accuracy: {:.4f}".format(eval_accuracy))
    
# Accuracy Quantized model
def evaluate_qmodel(model, test_loader, device, NBIT):
    
    disallowed_layer_names = []
    # linear operation, convolution, batchnorm
    whitelist=[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, nn.BatchNorm2d]
    whitelist_layer_types = tuple(whitelist)
    eligible_modules_list = []
    eligible_param_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, whitelist_layer_types) and name not in disallowed_layer_names:
            eligible_modules_list.append((name, mod))
            eligible_param_list.append(name)
    
    #1. extract weigth
    #2. quantized weight
    #3. Load quantized weight into the model
    for name, param in model.named_parameters():
        layername = '.'.join(name.split('.')[:1])
        layername2 = '.'.join(name.split('.')[:2])
        if layername in eligible_param_list:
            weight = param.cpu().detach().numpy()
            dqn = quantize(weight, NBIT)
            param.data = torch.from_numpy(dqn)
        elif layername2 in eligible_param_list:
            weight = param.cpu().detach().numpy()
            dqn = quantize(weight, NBIT)
            param.data = torch.from_numpy(dqn)
            

    #batch norm 고정, dropout 안함, gradient 계산안함
    model.eval()
    #model 파라미터를 지정한 device 메모리에 올림
    model.to(device)

    running_corrects = 0
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        #텐서의 최대값과 index를 반환
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # inputs.size(0) 현재 mini-batch의 크기(input의 0번째 dimension 크기)
        running_corrects += torch.sum(preds == labels.data)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    print("{}bit quantize model cifar100 Accuracy : {:.4f}".format(NBIT, eval_accuracy))

model = dongnet12()
model_dict = torch.load('model.pth', map_location=torch.device('cpu'))  # 상태 사전 로드
model.load_state_dict(model_dict)  # 모델에 상태 사전 로드

train_transform = transforms.Compose([
        #padding을 4추가하고 이미지를 32 X 32로 자름
        transforms.RandomCrop(32, padding=4),
        #이미지를 뒤집는다
        transforms.RandomHorizontalFlip(),
        #텐서로 바꾼다
        transforms.ToTensor(),
        #각 채널별 평균, 표준편차``
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR10 train 데이터 가져오기
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    # CIFAR10 test 데이터 가져오기
train_set = torchvision.datasets.CIFAR100(root="data", train=True, download=True, transform=train_transform) 
test_set = torchvision.datasets.CIFAR100(root="data", train=False, download=True, transform=test_transform)

train_sampler = torch.utils.data.RandomSampler(train_set)
test_sampler = torch.utils.data.SequentialSampler(test_set)

    #sampler은 dataset에서 데이터를 뽑아오는 역할, num_workers는 데이터를 읽어오는 프로세스 수
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=128,
    sampler=train_sampler, num_workers=8)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=128,
    sampler=test_sampler, num_workers=8)

# LOSS, Accuracy 확인
#evaluate_model(model=model, test_loader=test_loader, device='cuda')
evaluate_qmodel(model=model, test_loader=test_loader, device='cpu', NBIT=8)
torch.save(model.state_dict(), 'int8qmodel.pth')
#evaluate_qmodel(model=model, test_loader=test_loader, device='cuda', NBIT=7)
#evaluate_qmodel(model=model, test_loader=test_loader, device='cuda', NBIT=6)
#evaluate_qmodel(model=model, test_loader=test_loader, device='cuda', NBIT=5)
#evaluate_qmodel(model=model, test_loader=test_loader, device='cuda', NBIT=4)
#evaluate_qmodel(model=model, test_loader=test_loader, device='cuda', NBIT=3)
