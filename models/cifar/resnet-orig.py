from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math


__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, stepsize=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.stepsize = stepsize

    def forward(self, x):
        """
            x is not residual, but out...
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # there's no skip connection inside the block

        # the first block in a stage is downsampled to be consistent
        # should separate this out
        if self.downsample is not None:
            residual = self.downsample(x)

        # explicitly manipulate the residual
        out *= self.stepsize

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4 # channel expands by 4 times. Not good for dynamic view

    def __init__(self, inplanes, planes, stride=1, downsample=None, stepsize=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.stepsize = stepsize

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 
        out *= self.stepsize

        out += residual
        out = self.relu(out)

        return out

def isPowerOfTwo (x): 
    # check if an integer is power of two
    return (x and (not(x & (x - 1))) ) 

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock', growths=[1,1,1]):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            #todo
            assert False, 'should check the expansion of channel for this case'
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        # growing blocks
        for growth in growths:
            assert isPowerOfTwo(growth)

        # build the model
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n * growths[0])
        self.layer2 = self._make_layer(block, 32, n * growths[1], stride=2)
        self.layer3 = self._make_layer(block, 64, n * growths[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # initialize weights
        # todo: n is duplicate here
        # todo: send in last model, check the different layer based on growths, and copy the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # the first block for bottlenet is not equal: 16 - 16 x 4 for bottleneck
        # residual is built on output of previous stage, and add to the downsampled output of previous stage....
        # would be any difference if first downsample, no with skip connection, then continue with invariant block?
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # we don't need downsample in between, because inplanes are expanded.
            # inplanes = 64, outplanes = plane * 4 = 64, it's equal
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition(self):
        pass


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32, 16
        x = self.layer2(x)  # 16x16, 32
        x = self.layer3(x)  # 8x8, 64

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
