import torch
import torchvision.transforms as transforms

import os
import pandas as pd
from PIL import Image

"""
Usage: first follow the instructions in the readme to obtain the Udacity self-driving dataset.
Name the processed train, test sets as Ch2_train, Ch2_test and place them into a common folder named udacity-driving-dataset/
Then, place this script inside udacity-driving-dataset/ and run it :) 
"""


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
            timestamps = df['timestamp'].to_numpy()
            angles = df['angle'].to_numpy(dtype='float32')  # explicitly set to float32 to be compatible with PyTorch model output
        else:
            data_path = os.path.join(root, 'Ch2_test')
            df = pd.read_csv(os.path.join(data_path, 'CH2_final_evaluation.csv'))
            timestamps = df['frame_id'].to_numpy()
            angles = df['steering_angle'].to_numpy(dtype='float32')
        self.timestamps = timestamps
        self.angles = angles
        self.img_path = os.path.join(data_path, 'center')
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, f'{self.timestamps[idx]}.jpg'))
        target = self.angles[idx]
        img = img.resize(size=(200, 66), resample=Image.BILINEAR, box=(0, 200, 640, 480))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

# also returns the original image in addition to the resized/cropped one
class DrivingWithOriginalDataset(DrivingDataset):
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, f'{self.timestamps[idx]}.jpg'))
        img_resized = img.resize(size=(200, 66), resample=Image.BILINEAR, box=(0, 200, 640, 480))
        if self.transform is not None:
            img = self.transform(img)
            img_resized = self.transform(img_resized)
        target = self.angles[idx]
        return img, img_resized, target

def get_dataset():
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    train = DrivingDataset(root='.', train=True, transform=train_transform)
    test = DrivingDataset(root='.', train=False, transform=test_transform)
    return train, test


batch_size = 1
trainset, testset = get_dataset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

train_images = torch.empty((len(trainset), *img_shape))
idx = 0
for (image, label) in trainloader:
    train_images[idx] = image
    idx += 1
    if idx % 100 == 0:
        print(f'Thru {idx*batch_size}')
torch.save(train_images, 'driving_train.pt')

test_images = torch.empty((len(testset), *img_shape))
idx = 0
for (image, label) in testloader:
    test_images[idx] = image
    idx += 1
    if idx % 100 == 0:
        print(f'Thru {idx*batch_size}')
torch.save(test_images, 'driving_test.pt')
