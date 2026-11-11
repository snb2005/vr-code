import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import random

from .layers import *
import sys
sys.path.append('../')
from definitions import *

means = (0.4914, 0.4822, 0.4465)
stddevs = (0.2470, 0.2435, 0.2616) 
img_shape = (3, 32, 32)
num_classes = 10

def get_dataset(augment=True, get_train=True, get_val=False):
    if augment:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        )
    else:
        train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    
    if get_train:
        train = torchvision.datasets.CIFAR10(root='datasets/CIFAR10_Dataset', train=True, download=True, transform=train_transform)
        if get_val:
            train_size = int(len(train) * 0.8)
            indices = [k for k in range(len(train))]
            random.shuffle(indices)
            train = torch.utils.data.Subset(train, indices[:train_size])
            val = torchvision.datasets.CIFAR10(root='datasets/CIFAR10_Dataset', train=True, download=False, transform=test_transform)
            val = torch.utils.data.Subset(val, indices[train_size:])
            return train, val
        else:
            return train
    else:
        test = torchvision.datasets.CIFAR10(root='datasets/CIFAR10_Dataset', train=False, download=True, transform=test_transform)
        return test

class Network(nn.Module):
    """
    Factorized Residual Network (Factorized ResNet) for CIFAR-10 (Experiment 4).
    Features:
      - Conv Block 1: Initial 3x3 convolution (3 -> 32 channels).
      - Factorized Residual Block 1: Two sequential 3x3 convs (32 -> 32) with identity skip connection.
      - Factorized Residual Block 2: Downsampling 3x3 convs (32 -> 64) with 1x1 conv shortcut.
      - Spatial Alignment: AvgPool2d(2) to align spatial dimension to 8x8 (64 * 8 * 8 = 4096).
      - Classification Head: Linear(4096, 200) -> ReLU -> Linear(200, 10).
    Maintains backward compatibility with legacy 3-layer CNN checkpoints via `use_residual`.
    """
    def __init__(self, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])
        
        # Conv Block 1: Initial feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        
        if self.use_residual:
            # Factorized Residual Block 1 (32 -> 32 channels)
            self.res1_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.res1_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            
            # Factorized Residual Block 2 (32 -> 64 channels, downsampling)
            self.res2_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.res2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.res2_shortcut = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=False)
            
            # Spatial alignment to 8x8 (64 * 8 * 8 = 4096 features)
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            
            # Classification Head (200 hidden units)
            self.fc1 = nn.Linear(4096, 200)
            self.fc2 = nn.Linear(200, 10)
        else:
            # Baseline 3-layer CNN (Yang et al., 2023)
            self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
            self.fc1 = nn.Linear(4096, 150)
            self.fc2 = nn.Linear(150, 10)
    
    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.conv1(x))
        if self.use_residual:
            # Factorized Residual Block 1: out = F.relu(x + conv_block(x))
            x = F.relu(x + self.res1_conv2(F.relu(self.res1_conv1(x))))
            # Factorized Residual Block 2: out = F.relu(conv_path(x) + shortcut(x))
            conv_path = self.res2_conv2(F.relu(self.res2_conv1(x)))
            shortcut = self.res2_shortcut(x)
            x = F.relu(conv_path + shortcut)
            # Pool to 8x8: 64 * 8 * 8 = 4096 features
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x

FactorizedResNet = Network


gtss_train = [
    {
        ROTATE: (-10, 10, 0.001)
    },
    {
        ROTATE: (-2, 2, 0.02),
        SHEAR: (-0.02, 0.02, 0.0005)
    },
    {
        SCALE: (0.99, 1.01, 0.00005),
        ROTATE: (-1, 1, 0.005),
        CONTRAST: (0.99, 1.01, 0.005),
        BRIGHTNESS: (-0.001, 0.001, 0.002)
    }
]

gtss_certify = [
    {
        ROTATE: (-10, 10, 0.0002)
    },
    {
        ROTATE: (-2, 2, 0.01),
        SHEAR: (-0.02, 0.02, 0.00025)
    },
    {
        SCALE: (0.99, 1.01, 0.00005),
        ROTATE: (-1, 1, 0.005),
        CONTRAST: (0.99, 1.01, 0.005),
        BRIGHTNESS: (-0.001, 0.001, 0.002)
    }
]

# ABLATION CONFIGS
# gtss_train = [
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.015),
#         SHEAR: (-0.02, 0.02, 0.000375)
#     },
#     {
#         ROTATE: (-2, 2, 0.025),
#         SHEAR: (-0.02, 0.02, 0.000625)
#     },
#     {
#         ROTATE: (-2, 2, 0.03),
#         SHEAR: (-0.02, 0.02, 0.00075)
#     },
# ]

# gtss_certify = [
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
#     {
#         ROTATE: (-2, 2, 0.01),
#         SHEAR: (-0.02, 0.02, 0.00025)
#     },
# ]
