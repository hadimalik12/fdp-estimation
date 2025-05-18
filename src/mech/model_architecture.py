import torch.nn as nn

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

import torch.nn as nn

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
