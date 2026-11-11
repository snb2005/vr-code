import torch

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

import itertools
import numpy as np
import os
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm
import sys
sys.path.append('../')

from geometric.intervals import *
from geometric.interval_perturbations import *
from geometric.scalar_perturbations import *

import argparse
parser = argparse.ArgumentParser(description='Certify steering angle range wrt geometric transformations')
parser.add_argument('-b', '--batch-size', type=int, default=1024)
parser.add_argument('-p', '--network-path')
parser.add_argument('-c', '--config-num', type=int)
parser.add_argument('--val', action='store_true')
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--num-workers', type=int, default=0 if os.name == 'nt' else 4)
args = parser.parse_args()

batch_size = args.batch_size
network_path = args.network_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Network:', network_path, file=sys.stderr)


# LOAD DATASET AND TRANSFORMS
from configs.driving_config import *
if args.val:
    random.seed(args.seed)
    _, dataset = get_dataset(get_val=True)
else:
    dataset = get_dataset(get_train=False)

print('Number of images:', len(dataset), file=sys.stderr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

gts = gtss_certify[args.config_num]
gts_split = {}
for g, (theta_min, theta_max, interval_size) in gts.items():
    num_splits = round((theta_max - theta_min) / interval_size)
    intervals = []
    for k in range(num_splits):
        intervals.append((g, theta_min + k * interval_size, theta_min + (k+1) * interval_size))
    gts_split[g] = intervals
    print(f'{g}: {num_splits} splits', file=sys.stderr)
all_gts_splits = list(itertools.product(*gts_split.values()))  # stores all perturbation splits for whole specified range

print('Transforms:', file=sys.stderr)
has_geometric = False
has_pixelwise = False
for tfm in gts:
    if tfm in geometric_perturbations:
        has_geometric = True
    elif tfm in pixelwise_perturbations:
        has_pixelwise = True
    params = gts[tfm]
    print(f'{tfm}: [{params[0]}, {params[1]}], w={params[2]}', file=sys.stderr)


# LOAD NETWORK AND GET ORIGINAL STEERING ANGLES
net = Network(dropout=args.dropout, train=False).to(device)
net.load_state_dict(torch.load(network_path))
net.eval()

angles_pred = torch.zeros(len(dataset))
angles_gt = torch.zeros(len(dataset))
with torch.no_grad():
    for batch_i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        if inputs.size(0) < batch_size:
            angles_pred[batch_i * batch_size : ] = outputs
            angles_gt[batch_i * batch_size : ] = labels
        else:
            angles_pred[batch_i * batch_size : (batch_i + 1) * batch_size] = outputs
            angles_gt[batch_i * batch_size : (batch_i + 1) * batch_size] = labels

print('Computed scalar predictions', file=sys.stderr)


# CERTIFY ROBUSTNESS VIA INTERVAL BOUNDS
ibp_net = BoundedModule(net, torch.empty((batch_size, *img_shape)))
angles_min = torch.full((len(dataset),), 1e9)
angles_max = torch.full((len(dataset),), -1e9)
print('Certifying...', file=sys.stderr)

s_time = timer()

# precompute interpolation grids and pixelwise tfms
grids = [None for k in range(len(all_gts_splits))]
pwises = [None for k in range(len(all_gts_splits))]
for split_i, tfms in enumerate(tqdm(all_gts_splits)):
    interp_transforms = []
    pixelwise_transforms = []
    for (p_name, pmin, pmax) in tfms:
        if p_name in geometric_perturbations:
            interp_transforms.append((p_name, pmin, pmax))
        elif p_name in pixelwise_perturbations:
            pixelwise_transforms.append((p_name, pmin, pmax))
        else:
            raise ValueError('No such perturbation implemented!')
    if has_geometric:
        grids[split_i] = make_interp_grid(img_shape, interp_transforms)
    if has_pixelwise:
        pwises[split_i] = pixelwise_transforms

# start certification
with torch.no_grad():
    for batch_i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        for split_i, tfms in enumerate(tqdm(all_gts_splits)):
            interval_inputs = IntervalTensor(inputs, inputs)
            if has_geometric:
                interp_grid = grids[split_i]
                interval_inputs = spatial(interval_inputs, interp_grid)
            if has_pixelwise:
                pixelwise_transforms = pwises[split_i]
                interval_inputs = pixelwise(interval_inputs, pixelwise_transforms)

            inputs_L = interval_inputs.lower
            inputs_U = interval_inputs.upper
            ptb = PerturbationLpNorm(norm=np.inf, x_L=inputs_L, x_U=inputs_U)
            bounded_inputs = BoundedTensor(inputs, ptb)  # note: inputs will not actually be used here; instead the bounds inputs_L and inputs_U will be used directly
            lb, ub = ibp_net.compute_bounds(x=(bounded_inputs,), method="IBP")
            
            if inputs.size(0) < batch_size:
                angles_min[batch_i * batch_size : ] = torch.minimum(angles_min[batch_i * batch_size : ], lb.cpu())
                angles_max[batch_i * batch_size : ] = torch.maximum(angles_max[batch_i * batch_size : ], ub.cpu())
            else:
                angles_min[batch_i * batch_size : (batch_i + 1) * batch_size] = torch.minimum(angles_min[batch_i * batch_size : (batch_i + 1) * batch_size], lb.cpu())
                angles_max[batch_i * batch_size : (batch_i + 1) * batch_size] = torch.maximum(angles_max[batch_i * batch_size : (batch_i + 1) * batch_size], ub.cpu())

f_time = timer()
certify_time = f_time - s_time
print('Time:', certify_time, file=sys.stderr)

df = pd.DataFrame(data={
        'label': angles_gt.numpy(),
        'pred': angles_pred.numpy(),
        'cert_min': angles_min.numpy(),
        'cert_max': angles_max.numpy()
    }
)
os.makedirs('driving_certificates', exist_ok=True)
net_basename = os.path.splitext(os.path.basename(network_path))[0]
df.to_csv(f'driving_certificates/{net_basename}.csv', index=False)
