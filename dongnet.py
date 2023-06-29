import torch
import torch.nn as nn



class dongnet12(nn.Module):
    
    def __init__(self):
        super(dongnet12, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.leaky_relu5 = nn.LeakyReLU(negative_slope=0.1)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.leaky_relu6 = nn.LeakyReLU(negative_slope=0.1)   

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(256)
        self.leaky_relu7 = nn.LeakyReLU(negative_slope=0.1)   

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(256)
        self.leaky_relu8 = nn.LeakyReLU(negative_slope=0.1)

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm9 = nn.BatchNorm2d(256)
        self.leaky_relu9 = nn.LeakyReLU(negative_slope=0.1)

        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool2d = nn.MaxPool2d(2, stride=2)
        self.linear_relu = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 100)
        )
        #self.softmax = nn.Softmax(dim=1)

        #self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.batchnorm1(self.conv1(x))
        out = self.leaky_relu1(out)

        out = self.batchnorm2(self.conv2(out))
        out = self.leaky_relu2(out)
        out = self.max_pool2d(out)

        out = self.batchnorm3(self.conv3(out))
        out = self.leaky_relu3(out)        

        out = self.batchnorm4(self.conv4(out))
        out = self.leaky_relu4(out)

        out = self.batchnorm5(self.conv5(out))
        out = self.leaky_relu5(out)
        out = self.max_pool2d(out)

        out = self.batchnorm6(self.conv6(out))
        out = self.leaky_relu6(out)

        out = self.batchnorm7(self.conv7(out))
        out = self.leaky_relu7(out)

        out = self.batchnorm8(self.conv8(out))
        out = self.leaky_relu8(out)

        out = self.batchnorm9(self.conv9(out))
        out = self.leaky_relu9(out)
        out = self.max_pool2d(out)

        #out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear_relu(out)
        #out = self.softmax(out)
        
        return out
     