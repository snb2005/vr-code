import torch
import os

import math
import numpy as np
import random
from timeit import default_timer as timer
import sys
sys.path.append('../')

from geometric.intervals import *
from geometric.interval_perturbations import *

import argparse
parser = argparse.ArgumentParser(description='Compute per-pixel widths of interval bounds under geometric transformations')
parser.add_argument('-b', '--batch-size', type=int, default=512)
parser.add_argument('-d', '--dataset-name', choices=[MNIST, CIFAR10, TINY, DRIVING])
parser.add_argument('-t', '--dataset-type', choices=['train', 'test'])
parser.add_argument('-f', '--frac', type=float, default=1)     # fraction of dataset to compute
parser.add_argument('-s', '--samples', type=int, default=10)   # number of perturbation configurations to sample over
parser.add_argument('-c', '--config-num', type=int)
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--num-workers', type=int, default=0 if os.name == 'nt' else 16)
args = parser.parse_args()

batch_size = args.batch_size
dataset_name = args.dataset_name
dataset_type = args.dataset_type
samples = args.samples


# SEED THINGS
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(seed)
print(f'Seed set to {seed}', file=sys.stderr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD DATA
if dataset_name == MNIST:
    from configs.mnist_config import *
elif dataset_name == CIFAR10:
    from configs.cifar10_config import *
elif dataset_name == TINY:
    from configs.tinyimagenet_config import *
elif dataset_name == DRIVING:
    from configs.driving_config import *
else:
    raise ValueError('No such dataset!')

if dataset_type == 'train':
    gts = gtss_train[args.config_num]
else:
    gts = gtss_certify[args.config_num]

print('Transforms:')
for tfm in gts:
    params = gts[tfm]
    print(f'{tfm.capitalize()}: [{params[0]}, {params[1]}], w={params[2]}')

dataset = get_dataset(get_train=(dataset_type == 'train'))
num_images_to_compute = math.ceil(len(dataset) * args.frac)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


# CALCULATE INTERVAL OVER-APPROXIMATIONS FOR ALL PERTURBATION SPLITS
s_time = timer()
interval_widths_avg = []  # stores avg interval width for each sample
interval_widths_max = []  # stores max interval width for each sample

for sample in range(samples):
    tv = {
        ROTATE: 0.0,
        TRANSLATE_X: 0.0,
        TRANSLATE_Y: 0.0,
        SCALE: 1.0,
        SHEAR: 0.0,

        CONTRAST: 1.0,
        BRIGHTNESS: 0.0
    }
    
    interp_transforms = []
    pixelwise_transforms = []
    
    for (p_name, (pmin, pmax, interval_size)) in gts.items():
        sampled_tfm_val = random.uniform(pmin, pmax)
        if p_name == ROTATE or p_name == TRANSLATE_Y:
            tv[p_name] = -sampled_tfm_val
        elif p_name == SHEAR:
            tv[p_name] = np.rad2deg(np.arctan(sampled_tfm_val))  # unit conversion from % to degrees of shear
        else:
            tv[p_name] = sampled_tfm_val

        # interval bound checking to make sure interval does not exceed specified min/max bounds
        if sampled_tfm_val - interval_size/2 < pmin:
            tfm_interval = (pmin, pmin + interval_size)
        elif sampled_tfm_val + interval_size/2 > pmax:
            tfm_interval = (pmax - interval_size, pmax)
        else:
            tfm_interval = (sampled_tfm_val - interval_size/2, sampled_tfm_val + interval_size/2)

        if p_name in geometric_perturbations:
            interp_transforms.append((p_name, tfm_interval[0], tfm_interval[1]))
        elif p_name in pixelwise_perturbations:
            pixelwise_transforms.append((p_name, tfm_interval[0], tfm_interval[1]))
        else:
            raise ValueError('No such perturbation implemented!')
    n_interp_transforms = len(interp_transforms)
    n_pixelwise_transforms = len(pixelwise_transforms)
    
    if n_interp_transforms > 0:
        interp_grid = make_interp_grid(img_shape, interp_transforms)
    
    total = 0
    img_interval_widths_avg = 0.0
    img_interval_widths_max = 0.0
    for (inputs, _) in dataloader:
        inputs = inputs.to(device)
        inputs = IntervalTensor(inputs, inputs)

        if n_interp_transforms > 0:
            inputs = spatial(inputs, interp_grid)
        if n_pixelwise_transforms > 0:
            inputs = pixelwise(inputs, pixelwise_transforms)

        img_interval_widths_avg += (inputs.upper - inputs.lower).sum().item()
        img_interval_widths_max += torch.max((inputs.upper - inputs.lower).flatten(1), 1).values.sum().item()

        total += inputs.shape[0]
        if total >= num_images_to_compute:
            break

    avg_interval_width_avg = img_interval_widths_avg / total / (img_shape[0] * img_shape[1] * img_shape[2])
    avg_interval_width_max = img_interval_widths_max / total
    interval_widths_avg.append(avg_interval_width_avg)
    interval_widths_max.append(avg_interval_width_max)
    
    # print(f'{sample + 1} -- {interp_transforms} + {pixelwise_transforms} : w_avg={round(avg_interval_width_avg, 3)}, w_max={round(avg_interval_width_max, 3)}')

print('Total time:', timer() - s_time)
print('Average interval width (average of image pixels):', sum(interval_widths_avg) / len(interval_widths_avg))
print('Average interval width (max of image pixels):', sum(interval_widths_max) / len(interval_widths_max))
