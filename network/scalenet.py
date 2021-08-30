import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
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


class SABlock(nn.Module):
    layer_idx = 0
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, bias=False, downsample=False, structure=[], dilation=1):
        super(SABlock, self).__init__()

        channels = structure[SABlock.layer_idx][:-1]
        side = structure[SABlock.layer_idx][-1]
        SABlock.layer_idx += 1
        self.scales = [None, 2, 4, 7]
        self.stride = stride

        self.downsample = None if downsample == False else \
                          nn.Sequential(nn.Conv2d(inplanes, planes * SABlock.expansion, kernel_size=1, stride=1, bias=bias),
                                        nn.BatchNorm2d(planes * SABlock.expansion))

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)

        # kernel size == 1 if featuremap size == 1
        self.conv2 = nn.ModuleList([nn.Conv2d(planes, channels[i], kernel_size=3 if side / 2**i > 1 else 1, stride=1, padding=dilation if side / 2**i > 1 else 0, bias=bias, dilation=dilation if side / 2**i > 1 else 1) if channels[i] > 0 else \
                                    None for i in range(len(self.scales))])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(channels[i]) if channels[i] > 0 else \
                                  None for i in range(len(self.scales))])

        self.conv3 = nn.Conv2d(sum(channels), planes * SABlock.expansion, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(planes * SABlock.expansion)


    def forward(self, x):
        x = F.max_pool2d(x, self.stride, self.stride) if self.stride > 1 else x

        residual = self.downsample(x) if self.downsample != None else x

        out1 = self.conv1(x)
        out1 = F.relu(self.bn1(out1))

        out2_list = []
        size = [out1.size(2), out1.size(3)]
        for i in range(len(self.scales)):
            out2_i = out1 # copy
            if self.scales[i] != None:
                if out2_i.shape[3] < self.scales[i] or out2_i.shape[2] < self.scales[i]:
                    out2_i = F.max_pool2d(out2_i, out2_i.shape[2], out2_i.shape[3])
                else:
                    out2_i = F.max_pool2d(out2_i, self.scales[i], self.scales[i])
            if self.conv2[i] != None:
                out2_i = self.conv2[i](out2_i)
            if self.scales[i] != None:
                # nearest mode is not suitable for upsampling on non-integer multiples 
                mode = 'nearest' if size[0] % out2_i.shape[2] == 0 and size[1] % out2_i.shape[3] == 0 else 'bilinear'
                out2_i = F.upsample(out2_i, size=size, mode=mode)
            if self.bn2[i] != None:
                out2_i = self.bn2[i](out2_i)
                out2_list.append(out2_i)
        out2 = torch.cat(out2_list, 1)
        out2 = F.relu(out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 += residual
        out3 = F.relu(out3)

        return out3


class ScaleNet(nn.Module):

    def __init__(self, block, layers, structure, dilations=(1,1,2,4)):
        super(ScaleNet, self).__init__()

        self.inplanes = 64
        self.structure = structure

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=dilations[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.normalize = Normalize()
        self.not_training = []
        block.layer_idx = 0

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = True if stride != 1 or self.inplanes != planes * block.expansion else False
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, structure=self.structure, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, downsample=False, structure=self.structure, dilation=dilation))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 3, 2, 1)
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


def scalenet50(structure_path, ckpt=None, dilations=(1,1,1,1), **kwargs):
    layer = [3, 4, 6, 3]
    structure = json.loads(open(structure_path).read())
    model = ScaleNet(SABlock, layer, structure, dilations, **kwargs)

    # pretrained
    if ckpt != None:
        state_dict = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


def scalenet101(structure_path, ckpt=None, dilations=(1,1,1,1), **kwargs):
    layer = [3, 4, 23, 3]
    structure = json.loads(open(structure_path).read())
    model = ScaleNet(SABlock, layer, structure, dilations, **kwargs)

    # pretrained
    if ckpt != None:
        state_dict = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


def scalenet152(structure_path, ckpt=None, dilations=(1,1,1,1), **kwargs):
    layer = [3, 8, 36, 3]
    structure = json.loads(open(structure_path).read())
    model = ScaleNet(SABlock, layer, structure, dilations, **kwargs)

    # pretrained
    if ckpt != None:
        state_dict = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state_dict)

    return model

