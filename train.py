import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torch.optim as optim

import numpy as np
import os
import random
import sys
from timeit import default_timer as timer

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, BoundDataParallel
from torchattacks import PGD

from geometric.intervals import *
from geometric.interval_perturbations import *
from geometric.scalar_perturbations import *

import argparse
parser = argparse.ArgumentParser(description='Train robust networks against geometric transformations')
parser.add_argument('-d', '--dataset-name', choices=[MNIST, CIFAR10, TINY, DRIVING])
parser.add_argument('-t', '--train-mode', default=CGT_CROWN, choices=[AUGMENTED, PGD_A, IBP_A, CGT_IBP, CGT_CROWN, 'cgt_crown', 'cgtibp'])
parser.add_argument('--bound-method', default=None, choices=['IBP', 'CROWN-IBP'], help='Explicit bound propagation method (IBP or CROWN-IBP)')
parser.add_argument('--robust-batch-size', type=int, default=128, help='Sub-batch size for CROWN-IBP bounds propagation to prevent CUDA OOM on 6GB GPU')
parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
parser.add_argument('--tb-dir', default='runs', help='TensorBoard logging directory')
parser.add_argument('-c', '--config-num', type=int)
parser.add_argument('-b', '--batch-size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--epsilon', type=float)           # param only used in l-infinity methods
parser.add_argument('--save-every', type=int)          # save model every few epochs
parser.add_argument('--validate-every', type=int)
parser.add_argument('--save-dir')                      # where to save models
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--dropout', action='store_true')  # whether to use dropout for self-driving network
parser.add_argument('--multi-gpu', action='store_true')
parser.add_argument('--num-workers', type=int, default=0 if os.name == 'nt' else 4)
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint (.pt) to resume training from')
parser.add_argument('--start-epoch', type=int, default=None, help='Epoch index to resume from (0-indexed; auto-inferred if omitted)')
parser.add_argument('--start-interval-epoch', type=int, default=None, help='Epoch to start interval robust training (1-indexed)')
parser.add_argument('--ramp-up-epochs', type=int, default=None, help='Number of epochs to linearly increase perturbation interval size')
parser.add_argument('--milestones', type=int, nargs='+', default=None, help='Milestone epochs for lr decay')
parser.add_argument('--save-path', type=str, default=None, help='Explicit destination file path for trained model')
args = parser.parse_args()


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


# PARAMS
dataset_name = args.dataset_name
train_mode = args.train_mode
if train_mode == 'cgt_crown':
    train_mode = CGT_CROWN
elif train_mode == 'cgtibp':
    train_mode = CGT_IBP

if args.bound_method is not None:
    bound_method = args.bound_method
elif train_mode == CGT_CROWN:
    bound_method = 'CROWN-IBP'
else:
    bound_method = 'IBP'

batch_size = args.batch_size
num_epochs = args.epochs
lr = args.lr
linf_eps = args.epsilon


# TRAINING HYPERPARAMETERS
is_provable_mode = (train_mode in [CGT_IBP, CGT_CROWN, IBP_A] or 'ibp' in train_mode or 'cgt' in train_mode or 'crown' in train_mode)

if is_provable_mode:
    ramp_up_epochs = args.ramp_up_epochs if args.ramp_up_epochs is not None else (num_epochs / 2)  # number of epochs used to linearly increase eps from 0 to specified interval size
    kappa_f = 0.5                    # final weighting of interval loss term
    if args.start_interval_epoch is not None:
        start_interval_epoch = args.start_interval_epoch
    elif dataset_name == MNIST:
        start_interval_epoch = 16    # epoch at which to start using interval loss term
    elif dataset_name == CIFAR10:
        start_interval_epoch = 31
    elif dataset_name == TINY:
        start_interval_epoch = 31
    elif dataset_name == DRIVING:
        start_interval_epoch = 1
        if args.ramp_up_epochs is None:
            ramp_up_epochs = num_epochs
elif train_mode == PGD_A:
    if dataset_name == MNIST:
        pgd_alpha = 0.005
        pgd_steps = 40
    elif dataset_name == CIFAR10:
        pgd_alpha = 0.002
        pgd_steps = 7


# LOAD DATA AND INITIALIZE NETWORK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, file=sys.stderr)

if dataset_name == MNIST:
    from configs.mnist_config import *
elif dataset_name == CIFAR10:
    from configs.cifar10_config import *
elif dataset_name == TINY:
    from configs.tinyimagenet_config import *
elif dataset_name == DRIVING:
    from configs.driving_config import *

gts = gtss_train[args.config_num]
print('Transforms:', file=sys.stderr)
for tfm in gts:
    params = gts[tfm]
    if train_mode in [CGT_IBP, CGT_CROWN]:
        print(f'{tfm}: [{params[0]}, {params[1]}], w={params[2]}', file=sys.stderr)
    else:
        print(f'{tfm}: [{params[0]}, {params[1]}]', file=sys.stderr)

if args.validate_every is not None:
    trainset, valset = get_dataset(get_val=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
else:
    trainset = get_dataset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

if dataset_name == DRIVING and args.dropout:
    net = Network(dropout=True).to(device)
    print('Using dropout', file=sys.stderr)
else:
    net = Network().to(device)

is_res = getattr(net, 'use_residual', False)
if is_res:
    save_mode_str = "cgt_crown_resnet" if (train_mode == CGT_CROWN or bound_method == 'CROWN-IBP') else "cgt_resnet"
else:
    save_mode_str = "cgt_crown" if (train_mode == CGT_CROWN or bound_method == 'CROWN-IBP') else ("cgt" if train_mode == CGT_IBP else train_mode)

if args.save_path is not None:
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
else:
    os.makedirs(f'{args.save_dir}/{dataset_name}', exist_ok=True)
    save_path = f'{args.save_dir}/{dataset_name}/{save_mode_str}_'
    for g, (pmin, pmax, interval_size) in gts.items():
        if train_mode in [CGT_IBP, CGT_CROWN]:
            save_path += g + f'{pmin}^{pmax}^{interval_size}' + '_'
        else:
            save_path += g + f'{pmin}^{pmax}' + '_'
    if train_mode == IBP_A or train_mode == PGD_A:
        save_path += f'eps{linf_eps}_'
    save_path = save_path + f'seed{seed}.pt'

print(f'Network save path: {save_path}', file=sys.stderr)

optimizer = optim.Adam(net.parameters(), lr=lr)
if dataset_name == DRIVING:
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()

# learning rate scheduler
if args.milestones is not None:
    milestones = args.milestones
elif dataset_name == MNIST:
    milestones = [80]
elif dataset_name == CIFAR10:
    milestones = [100]
elif dataset_name == TINY:
    milestones = [120, 150]
elif dataset_name == DRIVING:
    milestones = [20, 40]

start_epoch = 0
if args.resume is not None:
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f'Checkpoint file not found: {args.resume}')
    print(f'Loading weights from {args.resume}...', file=sys.stderr)
    checkpoint = torch.load(args.resume, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    print(f'Successfully loaded weights from {args.resume}', file=sys.stderr)

    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    else:
        import re
        match = re.search(r'_ep(\d+)\.pt$', args.resume)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f'Auto-inferred start_epoch={start_epoch} (next is Epoch {start_epoch+1}) from filename.', file=sys.stderr)
        else:
            print('Could not auto-infer start_epoch from filename. Defaulting to 0. Use --start-epoch to override.', file=sys.stderr)
elif args.start_epoch is not None:
    start_epoch = args.start_epoch

for group in optimizer.param_groups:
    group.setdefault('initial_lr', group['lr'])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=start_epoch - 1 if start_epoch > 0 else -1)

if is_provable_mode:
    ibp_net = BoundedModule(net, torch.empty((1, *img_shape)))
    if args.multi_gpu:
        ibp_net = BoundDataParallel(ibp_net)
    kappa_step = (1 - kappa_f) / (num_epochs - start_interval_epoch + 1)
    kappa = 1.0

    if train_mode in [CGT_IBP, CGT_CROWN]:
        eps_step = {}
        eps = {}
        for p in gts:
            eps_step[p] = gts[p][2] / ramp_up_epochs
            eps[p] = 0.0
    else:
        eps_step = linf_eps / ramp_up_epochs
        eps = 0.0

    # Fast-forward schedules to start_epoch when resuming
    if start_epoch > 0:
        for e in range(start_epoch):
            if e + 1 >= start_interval_epoch:
                kappa -= kappa_step
                if e + 1 < start_interval_epoch + ramp_up_epochs:
                    if train_mode in [CGT_IBP, CGT_CROWN]:
                        for p in gts:
                            eps[p] += eps_step[p]
                    else:
                        eps += eps_step
        print(f'Resuming at epoch {start_epoch+1}: fast-forwarded kappa={kappa:.4f}', file=sys.stderr)
        if isinstance(eps, dict):
            for p in eps:
                print(f'{p} eps: {eps[p]}', file=sys.stderr)
        elif eps is not None:
            print(f'eps: {eps}', file=sys.stderr)
elif train_mode == PGD_A:
    atk = PGD(net, eps=linf_eps, alpha=pgd_alpha, steps=pgd_steps)

writer = None
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    tb_log_dir = os.path.join(args.tb_dir, f'{dataset_name}_{save_mode_str}_cfg{args.config_num}_seed{seed}')
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f'TensorBoard logging to {tb_log_dir}', file=sys.stderr)


# START TRAINING
total_time = 0
print(f'Start training {dataset_name} network with mode: {"cgt_crown" if (train_mode == CGT_CROWN or bound_method == "CROWN-IBP") else ("cgt" if train_mode == CGT_IBP else train_mode)} (bound_method={bound_method if is_provable_mode else "none"}) from epoch {start_epoch+1} to {num_epochs}', file=sys.stderr)

for epoch in range(start_epoch, num_epochs):
    print(f'**Epoch {epoch+1}**', file=sys.stderr)

    if args.save_every is not None and epoch % args.save_every == 0:
       torch.save(net.state_dict(), save_path[:-3] + f'_ep{epoch}.pt')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device=device)
    running_loss = 0.0
    running_clean_loss = 0.0
    running_robust_loss = 0.0

    if is_provable_mode and epoch + 1 >= start_interval_epoch:
        kappa -= kappa_step
        print(f'kappa: {kappa:.4f}', file=sys.stderr)

        if epoch + 1 < start_interval_epoch + ramp_up_epochs:
            if train_mode in [CGT_IBP, CGT_CROWN]:
                for p in gts:
                    eps[p] += eps_step[p]
                    print(f'{p} eps: {eps[p]}', file=sys.stderr)
            else:
                eps += eps_step
                print(f'eps: {eps}', file=sys.stderr)
    
    start = timer()

    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # augment images with a transform that is sampled within the perturbation ranges specified
        tv = {
            ROTATE: 0.0,
            TRANSLATE_X: 0.0,
            TRANSLATE_Y: 0.0,
            SCALE: 1.0,
            SHEAR: 0.0,
            CONTRAST: 1.0,
            BRIGHTNESS: 0.0,
        }

        if train_mode in [CGT_IBP, CGT_CROWN]:
            interp_transforms = []
            pixelwise_transforms = []
        
        for (p_name, (pmin, pmax, _)) in gts.items():
            sampled_tfm_val = random.uniform(pmin, pmax)
            if p_name == ROTATE or p_name == TRANSLATE_Y:
                tv[p_name] = -sampled_tfm_val
            elif p_name == SHEAR:
                tv[p_name] = np.rad2deg(np.arctan(sampled_tfm_val))  # unit conversion from % to degrees of shear (for PyTorch affine)
            else:
                tv[p_name] = sampled_tfm_val

            if train_mode in [CGT_IBP, CGT_CROWN]:
                # interval bound checking to make sure interval does not exceed specified min/max bounds
                if sampled_tfm_val - eps[p_name]/2 < pmin:
                    tfm_interval = (pmin, pmin + eps[p_name])
                elif sampled_tfm_val + eps[p_name]/2 > pmax:
                    tfm_interval = (pmax - eps[p_name], pmax)
                else:
                    tfm_interval = (sampled_tfm_val - eps[p_name]/2, sampled_tfm_val + eps[p_name]/2)

                if p_name in geometric_perturbations:
                    interp_transforms.append((p_name, tfm_interval[0], tfm_interval[1]))
                elif p_name in pixelwise_perturbations:
                    pixelwise_transforms.append((p_name, tfm_interval[0], tfm_interval[1]))
                else:
                    raise ValueError('No such perturbation implemented!')
        
        augmented_inputs = inputs
        if tv[ROTATE] != 0 or tv[TRANSLATE_X] != 0 or tv[TRANSLATE_Y] != 0 or tv[SCALE] != 1 or tv[SHEAR] != 0:
            augmented_inputs = Ft.affine(augmented_inputs, tv[ROTATE], (tv[TRANSLATE_X], tv[TRANSLATE_Y]), tv[SCALE], tv[SHEAR], interpolation=Ft.InterpolationMode.BILINEAR)
        if tv[CONTRAST] != 1 or tv[BRIGHTNESS] != 0:
            augmented_inputs = contrast_brightness(augmented_inputs, tv[CONTRAST], tv[BRIGHTNESS])

        if train_mode == AUGMENTED:
            outputs = net(augmented_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clean_loss_val = loss.item()
            robust_loss_val = 0.0
        elif is_provable_mode and epoch + 1 < start_interval_epoch:
            outputs = ibp_net(augmented_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clean_loss_val = loss.item()
            robust_loss_val = 0.0
        elif train_mode == PGD_A:
            adv_inputs = atk(augmented_inputs, labels)
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clean_loss_val = loss.item()
            robust_loss_val = 0.0
        elif is_provable_mode:
            # 1. Clean loss on geometrically augmented inputs
            outputs = ibp_net(augmented_inputs)
            clean_loss = criterion(outputs, labels)
            clean_loss_val = clean_loss.item()
            
            # Backprop clean loss scaled by kappa (if kappa > 0)
            if kappa > 0:
                (kappa * clean_loss).backward()

            robust_weight = 1.0 - kappa
            robust_loss_val = 0.0

            if robust_weight > 0:
                if bound_method == 'CROWN-IBP':
                    # Precompute FGV interpolation grid for current batch
                    if len(interp_transforms) > 0:
                        interp_grid = make_interp_grid(img_shape, interp_transforms)

                    r_bs = args.robust_batch_size if args.robust_batch_size is not None and args.robust_batch_size > 0 else inputs.size(0)
                    total_robust_sum = 0.0

                    # Chunking with robust_batch_size to manage VRAM under 6GB
                    for c_start in range(0, inputs.size(0), r_bs):
                        c_end = min(c_start + r_bs, inputs.size(0))
                        c_inputs = inputs[c_start:c_end]
                        c_labels = labels[c_start:c_end]
                        c_sz = c_inputs.size(0)

                        # FGV interval interpolation for current chunk
                        interval_chunk = IntervalTensor(c_inputs, c_inputs)
                        if len(interp_transforms) > 0:
                            interval_chunk = spatial(interval_chunk, interp_grid)
                        if len(pixelwise_transforms) > 0:
                            interval_chunk = pixelwise(interval_chunk, pixelwise_transforms)

                        ptb_chunk = PerturbationLpNorm(norm=np.inf, x_L=interval_chunk.lower, x_U=interval_chunk.upper)
                        bounded_chunk = BoundedTensor(c_inputs, ptb_chunk)

                        # Construct specification matrix C: f_y(x) - f_j(x) > 0
                        C_chunk = torch.zeros(c_sz, num_classes - 1, num_classes, device=device)
                        for i in range(c_sz):
                            y = c_labels[i]
                            C_chunk[i, :, y] = 1
                            j_idx = 0
                            for j in range(num_classes):
                                if j != y:
                                    C_chunk[i, j_idx, j] = -1
                                    j_idx += 1

                        # Differentiable CROWN-IBP bounds computation (gradients enabled)
                        lb_chunk, _ = ibp_net.compute_bounds(x=(bounded_chunk,), method='CROWN-IBP', C=C_chunk)

                        # Certified margin loss: L_robust = log(1 + sum_{j != y} exp(-m_j))
                        # Evaluated via logsumexp([0, -lb]) for exact numerical stability without overflow
                        zeros = torch.zeros(c_sz, 1, device=device)
                        chunk_margin_loss = torch.logsumexp(torch.cat([zeros, -lb_chunk], dim=1), dim=1).sum()

                        # Gradient accumulation: scale by (1 - kappa) / total_batch_size
                        scaled_chunk_loss = robust_weight * (chunk_margin_loss / inputs.size(0))
                        scaled_chunk_loss.backward()

                        total_robust_sum += chunk_margin_loss.item()

                    robust_loss_val = total_robust_sum / inputs.size(0)

                else:
                    # Baseline IBP training
                    if train_mode == IBP_A:
                        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
                    else:
                        interval_inputs = IntervalTensor(inputs, inputs)
                        if len(interp_transforms) > 0:
                            interp_grid = make_interp_grid(img_shape, interp_transforms)
                            interval_inputs = spatial(interval_inputs, interp_grid)
                        if len(pixelwise_transforms) > 0:
                            interval_inputs = pixelwise(interval_inputs, pixelwise_transforms)
                        ptb = PerturbationLpNorm(norm=np.inf, x_L=interval_inputs.lower, x_U=interval_inputs.upper)

                    bounded_inputs = BoundedTensor(augmented_inputs, ptb)
                    lb, ub = ibp_net.compute_bounds(x=(bounded_inputs,), method="IBP")

                    if dataset_name == DRIVING:
                        ibp_robust_loss = (criterion(lb, labels) + criterion(ub, labels)) / 2
                    else:
                        labels_onehot = F.one_hot(labels, num_classes=num_classes)
                        worst_case_outputs = lb * labels_onehot + ub * torch.logical_not(labels_onehot)
                        ibp_robust_loss = criterion(worst_case_outputs, labels)

                    (robust_weight * ibp_robust_loss).backward()
                    robust_loss_val = ibp_robust_loss.item()
        else:
            raise ValueError('No such training method!')

        nn.utils.clip_grad_norm_(net.parameters(), 8)
        optimizer.step()

        batch_total_loss = kappa * clean_loss_val + (1.0 - kappa) * robust_loss_val
        running_loss += batch_total_loss * inputs.size(0)
        running_clean_loss += clean_loss_val * inputs.size(0)
        running_robust_loss += robust_loss_val * inputs.size(0)
    
    end = timer()
    total_time += end - start

    max_memory = torch.cuda.max_memory_allocated(device=device) if device.type == 'cuda' else 0

    # run on val set
    val_statistic = None
    if args.validate_every is not None and (epoch + 1) % args.validate_every == 0:
        net.eval()
        correct = 0
        running_val_loss = 0
        with torch.no_grad():
            for (images, labels) in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = ibp_net(images)
                if dataset_name == DRIVING:
                    running_val_loss += criterion(outputs, labels).item() * images.size(0)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
        if dataset_name == DRIVING:
            val_statistic = round(running_val_loss / len(valset), 6)
        else:
            val_statistic = round(100 * correct / len(valset), 6)
        net.train()

    epoch_loss = round(running_loss / len(trainset), 6)
    epoch_clean_loss = round(running_clean_loss / len(trainset), 6)
    epoch_robust_loss = round(running_robust_loss / len(trainset), 6)
    epoch_time = round(end - start, 6)

    # TensorBoard logging
    if writer is not None:
        writer.add_scalar('Loss/total', float(epoch_loss), epoch + 1)
        writer.add_scalar('Loss/clean', float(epoch_clean_loss), epoch + 1)
        writer.add_scalar('Loss/robust', float(epoch_robust_loss), epoch + 1)
        writer.add_scalar('Schedule/kappa', float(kappa), epoch + 1)
        if isinstance(eps, dict):
            for p in eps:
                writer.add_scalar(f'Schedule/eps_{p}', float(eps[p]), epoch + 1)
        elif eps is not None:
            writer.add_scalar('Schedule/eps', float(eps), epoch + 1)
        if val_statistic is not None:
            writer.add_scalar('Accuracy/val', float(val_statistic), epoch + 1)

    # print info for current epoch
    if val_statistic is not None:
        print(f'Epoch {epoch+1} | Loss: {epoch_loss:.6f} (Clean: {epoch_clean_loss:.6f}, Robust: {epoch_robust_loss:.6f}, kappa: {kappa:.4f}) | Val {"Loss" if dataset_name == DRIVING else "Accuracy"}: {val_statistic} | Max Memory: {max_memory / (1024**2):.2f} MB | Time: {epoch_time:.2f}s', file=sys.stderr)
        print(f'{epoch+1},{epoch_time},{max_memory},{epoch_loss},{epoch_clean_loss},{epoch_robust_loss},{kappa},{val_statistic}')
    else:
        print(f'Epoch {epoch+1} | Loss: {epoch_loss:.6f} (Clean: {epoch_clean_loss:.6f}, Robust: {epoch_robust_loss:.6f}, kappa: {kappa:.4f}) | Max Memory: {max_memory / (1024**2):.2f} MB | Time: {epoch_time:.2f}s', file=sys.stderr)
        print(f'{epoch+1},{epoch_time},{max_memory},{epoch_loss},{epoch_clean_loss},{epoch_robust_loss},{kappa}')

    # step training parameters
    scheduler.step()
    if (epoch + 1) in milestones:
        print('lr:', scheduler.get_last_lr(), file=sys.stderr)

print(f'Finished training {dataset_name} network in {total_time} seconds', file=sys.stderr)
if args.save_path is not None:
    torch.save(net.state_dict(), args.save_path)
    print(f'Final model saved to {args.save_path}', file=sys.stderr)
else:
    final_pt = save_path[:-3] + '_final.pt'
    torch.save(net.state_dict(), final_pt)
    print(f'Final model saved to {final_pt}', file=sys.stderr)
    if getattr(net, 'use_residual', False):
        default_resnet_path = f'{args.save_dir}/{dataset_name}/cgt_crown_resnet_rotate-30^30^0.5.pt'
        torch.save(net.state_dict(), default_resnet_path)
        print(f'Also saved ResNet model to {default_resnet_path}', file=sys.stderr)
if writer is not None:
    writer.close()
