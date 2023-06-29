import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

from dongnet import dongnet12

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

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
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

def evaluate_model(model, test_loader, device, criterion=None):
    #batch norm 고정, dropout 안함, gradient 계산안함
    model.eval()
    #model 파라미터를 지정한 device 메모리에 올림
    model.to(device)

    running_loss = 0
    running_corrects = 0

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
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def train_model(model, train_loader, test_loader, device):

    # The training configurations were not carefully selected.
    learning_rate = 1e-2
    num_epochs = 93

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for epoch in range(num_epochs):

        # 트래인 모드로 변경
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #pytorch는 input을 batch 단위로 넣어서 backward pass하기에, 
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss는 배치의 평균 그래서 input.size(0)을 곱합
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model

def create_model():

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
#resnet.py에 kwargs가 되어있어서 이미의 파라미터 설정 가능    
    model = dongnet12()

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model

def main():
    random_seed = 0
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    set_random_seeds(random_seed=random_seed)

    model = create_model()

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)

    model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=cuda_device)

    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":

    main()