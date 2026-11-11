import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import random

from .layers import *
import sys
sys.path.append('../')
from definitions import *

means = (0.4819, 0.4457, 0.3934)
stddevs = (0.2734, 0.2650, 0.2770)
img_shape = (3, 56, 56)
num_classes = 200

def get_dataset(augment=True, get_train=True, get_val=False):
    if augment:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        )
    else:
        train_transform = transforms.Compose(
            [transforms.CenterCrop(56),
             transforms.ToTensor()]
        )
    test_transform = transforms.Compose(
        [transforms.CenterCrop(56),
         transforms.ToTensor()]
    )
    
    if get_train:
        train = torchvision.datasets.ImageFolder(root='datasets/tiny-imagenet-200/train', transform=train_transform)
        if get_val:
            train_size = int(len(train) * 0.8)
            indices = [k for k in range(len(train))]
            random.shuffle(indices)
            train = torch.utils.data.Subset(train, indices[:train_size])
            val = torchvision.datasets.ImageFolder(root='datasets/tiny-imagenet-200/train', transform=test_transform)
            val = torch.utils.data.Subset(val, indices[train_size:])
            return train, val
        else:
            return train
    else:
        test = torchvision.datasets.ImageFolder(root='datasets/tiny-imagenet-200/val', transform=test_transform)
        return test

from configs.wide_resnet_imagenet64 import wide_resnet_imagenet64
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])
        
        # WideResNet
        # self.model = wide_resnet_imagenet64()
        
        # CNN7
        in_ch = 3
        width = 64
        linear_size = 512
        
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25088, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, 200)
        )
    
    def forward(self, x):
        x = self.normalize(x)
        x = self.model(x)
        return x


gtss_train = [
    {
        ROTATE: (-5, 5, 0.005)
    },
    {
        SCALE: (0.98, 1.02, 0.0001)
    },
    {
        SHEAR: (-0.02, 0.02, 0.0001)
    }
]

gtss_certify = [
    {
        ROTATE: (-5, 5, 0.001)
    },
    {
        SCALE: (0.98, 1.02, 0.00002)
    },
    {
        SHEAR: (-0.02, 0.02, 0.00002)
    }
]

# ABLATION CONFIGS
# gtss_train = [
#     {
#         SCALE: (0.98, 1.02, 0.00002)
#     },
#     {
#         SCALE: (0.98, 1.02, 0.00006)
#     },
#     {
#         SCALE: (0.98, 1.02, 0.00014)
#     },
#     {
#         SCALE: (0.98, 1.02, 0.00018)
#     },
# ]

# gtss_certify = [
#     {
#         SCALE: (0.98, 1.02, 0.00002)
#     },
#     {
#         SCALE: (0.98, 1.02, 0.00002)
#     },
#     {
#         SCALE: (0.98, 1.02, 0.00002)
#     },
#     {
#         SCALE: (0.98, 1.02, 0.00002)
#     },
# ]
