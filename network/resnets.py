import torch.nn as nn 
import torch 
import sys 
import numpy as np

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img
 
class BasicBlock(nn.Module): 
    expansion = 1 
 
    def __init__(self, inplanes, planes, stride=1, downsample=False): 
        super(BasicBlock, self).__init__() 
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, 3, 1, 1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes * self.expansion) 
        if downsample: 
            self.downsample = nn.Sequential( 
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride, bias=False), 
                nn.BatchNorm2d(planes * self.expansion), 
            ) 
        else: 
            self.downsample = nn.Sequential() 
 
    def forward(self, x): 
        residual = self.downsample(x) 
 
        out = self.conv1(x) 
        out = self.bn1(out) 
        out = self.relu(out) 
 
        out = self.conv2(out) 
        out = self.bn2(out) 
 
        out += residual 
        out = self.relu(out) 
 
        return out 
 
 
class Bottleneck(nn.Module): 
    expansion = 4 
 
    def __init__(self, inplanes, planes, stride=1, downsample=False, dilation=1): 
        super(Bottleneck, self).__init__() 
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, bias=False, dilation=dilation) 
        self.bn2 = nn.BatchNorm2d(planes) 
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) 
        self.relu = nn.ReLU(inplace=True) 
        if downsample: 
            self.downsample = nn.Sequential( 
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride, bias=False), 
                nn.BatchNorm2d(planes * self.expansion), 
            ) 
        else: 
            self.downsample = nn.Sequential() 
 
    def forward(self, x): 
        residual = self.downsample(x) 
 
        out = self.conv1(x) 
        out = self.bn1(out) 
        out = self.relu(out) 
 
        out = self.conv2(out) 
        out = self.bn2(out) 
        out = self.relu(out) 
 
        out = self.conv3(out) 
        out = self.bn3(out) 
 
        out += residual 
        out = self.relu(out) 
 
        return out 
 
 
def init_weight(module): 
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu') 
 
 
class ResNet(nn.Module): 
 
    def __init__(self, block=Bottleneck, layers=[3, 4, 23, 3], base_channels=64): 
        self.inplanes = base_channels 
        super(ResNet, self).__init__() 
        self.conv1 = nn.Conv2d(3, base_channels, 7, 2, 3, bias=False) 
        #self.conv1 = nn.Conv2d(3, base_channels, 3, 2, 1, bias=False) 
        self.bn1 = nn.BatchNorm2d(base_channels) 
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) 
        self.layer1 = self._make_layer(block, base_channels * 1, layers[0]) 
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], dilation=2) 
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], dilation=4) 
 
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): 
                init_weight(m) 
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1) 
                nn.init.constant_(m.bias, 0) 

        self.normalize = Normalize()
        self.not_training = []

 
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1): 
        layers = [] 
        downsample = stride != 1 or self.inplanes != planes * block.expansion 
        layers.append(block(self.inplanes, planes, stride, downsample, dilation)) 
        self.inplanes = planes * block.expansion 
        for _ in range(1, blocks): 
            layers.append(block(self.inplanes, planes, dilation=dilation)) 
 
        return nn.Sequential(*layers) 
 
    def forward(self, x): 
        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.maxpool(x) 
 
        x = self.layer1(x) 
        x2 = self.layer2(x) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 

        return dict({'x2': x2, 'x3': x3, 'x4': x4})

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for p in layer.parameters():
                    p.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return
