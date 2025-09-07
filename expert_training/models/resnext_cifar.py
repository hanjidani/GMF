import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedConv2d(nn.Module):
    """Grouped convolution for ResNeXt"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(GroupedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        
    def forward(self, x):
        return self.conv(x)


class ResNeXtBottleneck(nn.Module):
    """ResNeXt Bottleneck Block"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, cardinality=32, base_width=4, drop_rate=0.0):
        super(ResNeXtBottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * cardinality
        
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        self.conv2 = GroupedConv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32, base_width=4, num_classes=100, drop_rate=0.3):
        super(ResNeXt, self).__init__()
        self.in_planes = 64
        self.cardinality = cardinality
        self.base_width = base_width
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_rate=drop_rate)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, drop_rate=0.0):
        layers = []
        layers.append(block(self.in_planes, planes, stride, self.cardinality, self.base_width, drop_rate))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, cardinality=self.cardinality, 
                              base_width=self.base_width, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        logits = self.fc(features)
        
        return features, logits


def resnext29_8x64d(num_classes=100, drop_rate=0.3):
    """ResNeXt-29, 8x64d for CIFAR-100
    Often achieves 83-86% accuracy on CIFAR-100
    """
    return ResNeXt(ResNeXtBottleneck, [3, 3, 3], cardinality=8, base_width=64, 
                   num_classes=num_classes, drop_rate=drop_rate)


def resnext29_16x64d(num_classes=100, drop_rate=0.3):
    """ResNeXt-29, 16x64d for CIFAR-100
    Higher cardinality version, potentially better performance
    """
    return ResNeXt(ResNeXtBottleneck, [3, 3, 3], cardinality=16, base_width=64, 
                   num_classes=num_classes, drop_rate=drop_rate)


class PreActResNeXtBottleneck(nn.Module):
    """Pre-activation ResNeXt Bottleneck - often performs better"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, cardinality=32, base_width=4, drop_rate=0.0):
        super(PreActResNeXtBottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * cardinality
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(width)
        self.conv2 = GroupedConv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        
        self.bn3 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv3(self.relu(self.bn3(out)))
        out += self.shortcut(x)
        return out


class PreActResNeXt(nn.Module):
    """Pre-activation ResNeXt"""
    def __init__(self, block, layers, cardinality=32, base_width=4, num_classes=100, drop_rate=0.3):
        super(PreActResNeXt, self).__init__()
        self.in_planes = 64
        self.cardinality = cardinality
        self.base_width = base_width
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_rate=drop_rate)
        
        self.bn = nn.BatchNorm2d(256 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, drop_rate=0.0):
        layers = []
        layers.append(block(self.in_planes, planes, stride, self.cardinality, self.base_width, drop_rate))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, cardinality=self.cardinality, 
                              base_width=self.base_width, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        logits = self.fc(features)
        
        return features, logits


def preact_resnext29_8x64d(num_classes=100, drop_rate=0.3):
    """Pre-activation ResNeXt-29, 8x64d for CIFAR-100
    Often the best performing variant
    """
    return PreActResNeXt(PreActResNeXtBottleneck, [3, 3, 3], cardinality=8, base_width=64, 
                         num_classes=num_classes, drop_rate=drop_rate)
