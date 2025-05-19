import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.gn1 = nn.GroupNorm(4, planes)  # 4 groups for normalization
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(4, planes)  # 4 groups for normalization
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(4, self.expansion * planes)  # 4 groups for normalization
            )

    def forward(self, x):
        out = self.leaky_relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.leaky_relu(out)
        return out

def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    ) 

def convnet_balanced(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(4, 32),  # 4 groups for 32 channels
        nn.LeakyReLU(0.01, inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, 64),
        nn.LeakyReLU(0.01, inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, 64),
        nn.LeakyReLU(0.01, inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, 128),
        nn.LeakyReLU(0.01, inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),

        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes),
    )

def resnet20(num_classes):
    class ResNet20(nn.Module):
        def __init__(self, num_classes):
            super(ResNet20, self).__init__()
            self.in_planes = 16

            self.conv1 = conv3x3(3, 16)
            self.gn1 = nn.GroupNorm(4, 16)  # 4 groups for 16 channels
            self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)
            self.layer1 = self._make_layer(16, 3, stride=1)
            self.layer2 = self._make_layer(32, 3, stride=2)
            self.layer3 = self._make_layer(64, 3, stride=2)
            self.linear = nn.Linear(64, num_classes)

        def _make_layer(self, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(BasicBlock(self.in_planes, planes, stride))
                self.in_planes = planes * BasicBlock.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.leaky_relu(self.gn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = nn.AdaptiveAvgPool2d((1, 1))(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
    
    return ResNet20(num_classes)

# List of supported neural network architectures
MODEL_MAPPING = {
    'convnet': convnet,
    'convnet_balanced': convnet_balanced,
    'resnet20': resnet20
}
