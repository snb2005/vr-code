import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torchvision.transforms.functional as Ft

import os
import pandas as pd
import random

from .layers import *
import sys
sys.path.append('../')
from definitions import *

means = (0.2779, 0.3013, 0.3703)
stddevs = (0.2274, 0.2265, 0.2600)
img_shape = (3, 66, 200)

class DrivingDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(DrivingDataset).__init__()
        
        if train:
            data_path = os.path.join(root, 'Ch2_train')
            df = pd.read_csv(os.path.join(data_path, 'interpolated.csv'))
            df = df[df['frame_id'] == 'center_camera']
            angles = df['angle'].to_numpy(dtype='float32')  # explicitly set to float32 to be compatible with PyTorch model output
        else:
            data_path = os.path.join(root, 'Ch2_test')
            df = pd.read_csv(os.path.join(data_path, 'CH2_final_evaluation.csv'))
            angles = df['steering_angle'].to_numpy(dtype='float32')
        
        self.angles = angles
        self.transform = transform
        self.train = train

        if train:
            self.preloaded_images = torch.load(os.path.join(root, 'driving_train.pt'))
        else:
            self.preloaded_images = torch.load(os.path.join(root, 'driving_test.pt'))
    
    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        img = self.preloaded_images[idx]
        target = self.angles[idx]

        if self.train and torch.rand(1) < 0.5:
            img = Ft.hflip(img)
            target = -target
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

def get_dataset(get_train=True, get_val=False):
    if get_train:
        train = DrivingDataset(root='datasets/udacity-driving-dataset', train=True)
        if get_val:
            train_size = int(len(train) * 0.8)
            indices = [k for k in range(len(train))]
            random.shuffle(indices)
            train = torch.utils.data.Subset(train, indices[:train_size])
            val = DrivingDataset(root='datasets/udacity-driving-dataset', train=True)
            val = torch.utils.data.Subset(val, indices[train_size:])
            return train, val
        else:
            return train
    else:
        test = DrivingDataset(root='datasets/udacity-driving-dataset', train=False)
        return test

class Network(nn.Module):
    def __init__(self, dropout=False, train=True):
        super().__init__()
        self.normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        if dropout:
            if train:
                self.fc_layers = nn.Sequential(
                    nn.Linear(1152, 100), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(100, 50), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(50, 10), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(10, 1)
                )
            else:
                self.fc_layers = nn.Sequential(
                    nn.Linear(1152, 100), nn.ReLU(), nn.Identity(),
                    nn.Linear(100, 50), nn.ReLU(), nn.Identity(),
                    nn.Linear(50, 10), nn.ReLU(), nn.Identity(),
                    nn.Linear(10, 1)
                )
        else:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152, 100), nn.ReLU(),
                nn.Linear(100, 50), nn.ReLU(),
                nn.Linear(50, 10), nn.ReLU(),
                nn.Linear(10, 1)
            )
    
    def forward(self, x):
        x = self.normalize(x)
        x = self.fc_layers(self.conv_layers(x))
        return x.view(-1)


gtss_train = [
    {
        ROTATE: (-2, 2, 0.004)
    }
]

gtss_certify = [
    {
        ROTATE: (-2, 2, 0.001)
    }
]
