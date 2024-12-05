import torch
import torch.nn as nn
import torch.nn.functional as F


# Bottleneck
"""
Bottleneck 구조:
1. 입력 채널(in_channels)를 1x1 conv를 통해 축소 -> 계산 효율성 증가
2. 축소된 채널 (bottleneck_channels)에서 다시 3x3 conv를 통해 새로운 피처맵 생성.

Channel Concatenation:
새로 생성된 피처맵(new_features)를 입력x와 concatenate하여 출력. -> 네트워크는 이전
모든 레이어의 출력을 활용

Grow Rate(k):
growth_rate는 새로 추가되는 피처맵의 채널 수이며 k=32라면 각 레이어가 32개 새로운 채널을 생성 
"""
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate):
        super(BottleneckLayer, self).__init__()
        bottleneck_channels = growth_rate * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                               kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        bottleneck_output = self.conv1(F.relu(self.bn1(x)))
        new_features = self.conv2(F.relu(self.bn2(bottleneck_output)))
        return torch.cat([x, new_features], dim=1)


# DenseBlock
"""
DenseBlock 구조:
1. 여러개의 BottleneckLayer가 쌓인 구조
2. num_layers: DenseBlock에 포함될 layer 수
3. 각 레이어는 입력 체널의 증가(in_channels + i * growth_rate)

ModuleList:
Pytorch에서 여러 레이러를 반복적으로 호출하기 위해 사용
"""
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            BottleneckLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#Transition Layer
"""
1x1 conv:
채널 수를 줄이는 역할

Average Pooling:
특성맵의 크기를 절반으로 줄이기 위해 사용 (stride=2)

Transition Layer의 필요성:
DenseBlock에서 채널 수가 계속 증가하므로, 
Transition Layer를 통해 채널 수를 줄이고 계산량을 조절
"""
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        return self.pool(x)


# DenseNet model
class DenseNet(nn.Module):
    def __init__(self, block_config, growth_rate=32, num_classes=100):
        super(DenseNet, self).__init__()

        num_channels = 2 * growth_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_channels, growth_rate)
            self.features.append(block)
            num_channels += num_layers * growth_rate

            if i != len(block_config) -1:
                transition = TransitionLayer(num_channels, num_channels // 2)
                self.features.append(transition)
                num_channels = num_channels // 2

        self.bn_final = nn.BatchNorm2d(num_channels)

        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)

        for layer in self.features:
            x = layer(x)

        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def densenet121(num_classes=100):
    return DenseNet(growth_rate=32, block_config=[6, 12, 24, 16], num_classes=num_classes)

def densenet169(num_classes=100):
    return DenseNet(growth_rate=32, block_config=[6, 12, 32, 32], num_classes=num_classes)

def densenet201(num_classes=100):
    return DenseNet(growth_rate=32, block_config=[6, 12, 48, 32], num_classes=num_classes)

def densenet264(num_classes=100):
    return DenseNet(growth_rate=32, block_config=[6, 12, 64, 48], num_classes=num_classes)