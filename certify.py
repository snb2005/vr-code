import torch
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, BoundDataParallel

import gc
import itertools
import numpy as np
import os
import subprocess
from timeit import default_timer as timer
from tqdm import tqdm
import sys
sys.path.append('../')

from geometric.intervals import *
from geometric.interval_perturbations import *
from geometric.scalar_perturbations import *

import argparse
parser = argparse.ArgumentParser(description='Certify robustness of networks against geometric transformations')
parser.add_argument('-b', '--batch-size', type=int, default=1024, help='Batch size for clean accuracy evaluation')
parser.add_argument('--eval-batch-size', type=int, default=128, help='Max images per bound-propagation batch (halved automatically on CUDA OOM)')
parser.add_argument('--method', default='CROWN-IBP', choices=['CROWN-IBP', 'CROWN', 'IBP', 'IBP-C'], help='Bound propagation method: CROWN-IBP, CROWN (tighter/slower), IBP, or IBP-C')
parser.add_argument('-d', '--dataset-name', choices=[MNIST, CIFAR10, TINY])
parser.add_argument('-p', '--network-path')
parser.add_argument('-c', '--config-num', type=int)
parser.add_argument('--save', action='store_true')
parser.add_argument('--val', action='store_true')
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--multi-gpu', action='store_true', help='(not recommended) wrap in BoundDataParallel; prefer --num-gpus')
parser.add_argument('--num-workers', type=int, default=0 if os.name == 'nt' else 2)
parser.add_argument('--max-images', type=int, default=None, help='Optionally limit number of images to certify')
parser.add_argument('--init-cells', type=int, default=0, help='Width (in original cells) of the first boxes tried; 0 = whole range as a single box (recommended: the splitting finds the right granularity itself)')
parser.add_argument('--num-gpus', type=int, default=1, help='Run this many worker processes, one per GPU, each certifying a disjoint share of the images')
parser.add_argument('--shard-id', type=int, default=None, help=argparse.SUPPRESS)  # internal: set by the launcher
parser.add_argument('--ckpt-dir', default='ckpt', help='Where checkpoints / per-GPU results are written')
parser.add_argument('--ckpt-interval', type=int, default=300, help='Seconds between checkpoints')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoints in --ckpt-dir (finished GPUs are skipped)')
parser.add_argument('--profile', action='store_true', help='Report time spent building interpolation grids vs. computing bounds')
parser.add_argument('--falsify-points', type=int, default=0, help='Before BaB, test this many concrete transformation parameters per image and drop images that are misclassified (sound: such images can never be certified)')
parser.add_argument('--max-node-evals', type=int, default=0, help='Reject (conservatively) an image after it took part in this many BaB node evaluations; 0 = unlimited')
parser.add_argument('--min-cells', type=int, default=1, help='Boxes this many cells wide (or less) are never split further; if they fail, the image is rejected. 1 = original behaviour')
parser.add_argument('--time-limit', type=float, default=0, help='Wall-clock seconds per GPU worker; when hit, every image not fully certified so far is counted as NOT certified. 0 = unlimited')
args = parser.parse_args()

batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
cert_method = args.method
dataset_name = args.dataset_name
network_path = args.network_path
net_filename = os.path.basename(network_path)
run_tag = f'{dataset_name}_{net_filename}_{cert_method}_c{args.config_num}_n{args.max_images}_v{int(args.val)}'
os.makedirs(args.ckpt_dir, exist_ok=True)


def shard_result_path(k, n):
    return os.path.join(args.ckpt_dir, f'{run_tag}_shard{k}of{n}.result.pt')


def shard_ckpt_path(k, n):
    return os.path.join(args.ckpt_dir, f'{run_tag}_shard{k}of{n}.ckpt')


def report_and_save(robust_indices, num_correct, dataset_len, certify_time, max_gpu_memory_bytes):
    num_robust = len(robust_indices)
    certified_accuracy_pct = round(100.0 * num_robust / dataset_len, 4)
    clean_accuracy_pct = round(100.0 * num_correct / dataset_len, 4)
    total_cert_time_s = round(certify_time, 4)
    print(f'Clean Correct: {num_correct} / {dataset_len} ({clean_accuracy_pct}%)', file=sys.stderr)
    print(f'Certified Robust ({cert_method}): {num_robust} / {dataset_len} ({certified_accuracy_pct}%)', file=sys.stderr)
    print(f'Total Cert Time: {total_cert_time_s:.4f} s', file=sys.stderr)
    print(f'Max GPU Memory: {max_gpu_memory_bytes} bytes ({max_gpu_memory_bytes / (1024**2):.2f} MB)', file=sys.stderr)
    print(f'{network_path},{cert_method},{num_correct},{clean_accuracy_pct},{num_robust},{certified_accuracy_pct},{total_cert_time_s},{max_gpu_memory_bytes}')
    if args.save:
        os.makedirs(f'certified_indices/{dataset_name}', exist_ok=True)
        torch.save(robust_indices, f'certified_indices/{dataset_name}/indices_{cert_method.lower()}_{net_filename}')


# ---------------------------------------------------------------------------------------------
# LAUNCHER: one worker process per GPU, each certifying a disjoint, interleaved share of the images.
# Images are independent, so sharding does not change which images get certified.
# ---------------------------------------------------------------------------------------------
if args.num_gpus > 1 and args.shard_id is None:
    n = args.num_gpus
    vis = os.environ.get('CUDA_VISIBLE_DEVICES')
    devs = vis.split(',') if vis else [str(k) for k in range(n)]
    if len(devs) < n:
        raise SystemExit(f'--num-gpus {n} requested but only {len(devs)} GPU(s) visible')
    if not args.resume:
        for k in range(n):
            for p in (shard_result_path(k, n), shard_ckpt_path(k, n)):
                if os.path.exists(p):
                    os.remove(p)
    t0 = timer()
    procs = []
    for k in range(n):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = devs[k]
        cmd = [sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:] + ['--shard-id', str(k)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        raise SystemExit(f'Worker exit codes: {codes}. Re-run the same command with --resume to continue from checkpoints.')
    results = [torch.load(shard_result_path(k, n)) for k in range(n)]
    robust_indices = sorted(i for r in results for i in r['robust_indices'])
    report_and_save(robust_indices, results[0]['num_correct'], results[0]['dataset_len'],
                    max(r['cert_time'] for r in results), max(r['max_mem'] for r in results))
    print(f'Wall-clock incl. data loading: {timer() - t0:.1f} s', file=sys.stderr)
    sys.exit(0)

standalone = args.shard_id is None
shard_id = 0 if standalone else args.shard_id
num_shards = 1 if standalone else args.num_gpus

if args.resume and os.path.exists(shard_result_path(shard_id, num_shards)):
    print(f'Shard {shard_id} already finished, skipping.', file=sys.stderr)
    if standalone:
        r = torch.load(shard_result_path(0, 1))
        report_and_save(r['robust_indices'], r['num_correct'], r['dataset_len'], r['cert_time'], r['max_mem'])
    sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[shard {shard_id}/{num_shards}] Network:', network_path, file=sys.stderr)


# LOAD DATASET AND TRANSFORMS
if dataset_name == MNIST:
    from configs.mnist_config import *
elif dataset_name == CIFAR10:
    from configs.cifar10_config import *
elif dataset_name == TINY:
    from configs.tinyimagenet_config import *
else:
    raise ValueError('No such dataset!')

if args.val:
    random.seed(args.seed)
    _, dataset = get_dataset(get_val=True)
else:
    dataset = get_dataset(get_train=False)

if args.max_images is not None:
    dataset = torch.utils.data.Subset(dataset, range(min(args.max_images, len(dataset))))

dataset_len = len(dataset)
print('Number of images:', dataset_len, file=sys.stderr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

gts = gtss_certify[args.config_num]
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


# LOAD NETWORK
checkpoint = torch.load(network_path, map_location=device)
state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
has_residual = ('downsample.weight' in state_dict or
                'res1_conv1.weight' in state_dict or
                'res2_shortcut.weight' in state_dict or
                'resnet' in network_path.lower())

try:
    net = Network(use_residual=has_residual).to(device)
except TypeError:
    net = Network().to(device)
net.load_state_dict(state_dict)
net.eval()


# CLEAN PASS: find correctly classified images AND keep them, so the dataset is only loaded once
correct_chunks, kept_inputs, kept_labels = [], [], []
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = net(inputs.to(device, non_blocking=True))
        _, predicted = torch.max(outputs.data, 1)
        ok = (predicted.cpu() == labels)
        correct_chunks.append(ok)
        if ok.any():
            kept_inputs.append(inputs[ok])
            kept_labels.append(labels[ok])

correct = torch.cat(correct_chunks)
num_correct = correct.sum().item()
print('Correct:', num_correct, file=sys.stderr)
correct_indices = correct.nonzero().flatten()

if num_correct > 0:
    cx = torch.cat(kept_inputs)
    cy = torch.cat(kept_labels)
else:
    cx = torch.empty((0, *img_shape))
    cy = torch.empty((0,), dtype=torch.long)
del kept_inputs, kept_labels, correct_chunks

# this worker's share of the correctly classified images (interleaved for load balance)
sel = torch.arange(shard_id, num_correct, num_shards)
cx, cy, shard_global = cx[sel], cy[sel].to(device), correct_indices[sel]
n_img = sel.numel()
print(f'[shard {shard_id}] certifying {n_img} images', file=sys.stderr)

# keep images on the GPU if they fit comfortably, otherwise in pinned CPU memory
cx_on_gpu = device.type == 'cuda' and cx.numel() * cx.element_size() <= 2 * 1024 ** 3
if cx_on_gpu:
    cx = cx.to(device)
elif device.type == 'cuda':
    cx = cx.pin_memory()


def fetch_inputs(idx):
    """idx: LongTensor on `device` with positions into this shard's image tensor."""
    if cx_on_gpu or device.type == 'cpu':
        return cx[idx]
    return cx[idx.cpu()].to(device, non_blocking=True)


def make_spec_matrix(labels, num_classes):
    """Vectorised C with C[i, r, y_i] = 1 and C[i, r, j_r] = -1 for the r-th class j_r != y_i (ascending order)."""
    B = labels.size(0)
    cls = torch.arange(num_classes, device=labels.device)
    others = cls.unsqueeze(0).expand(B, num_classes)[cls.unsqueeze(0) != labels.unsqueeze(1)].view(B, num_classes - 1)
    C = torch.zeros(B, num_classes - 1, num_classes, device=labels.device)
    C.scatter_(2, others.unsqueeze(2), -1.0)
    C.scatter_(2, labels.view(B, 1, 1).expand(B, num_classes - 1, 1), 1.0)
    return C


# ---------------------------------------------------------------------------------------------
# ADAPTIVE (BRANCH-AND-BOUND) INTERVAL SPLITTING
# ---------------------------------------------------------------------------------------------
ibp_net = BoundedModule(net, torch.empty((eval_batch_size, *img_shape)))
if args.multi_gpu:
    ibp_net = BoundDataParallel(ibp_net)

dims = list(gts.keys())
dim_params = [gts[g] for g in dims]
dim_n = [round((tmax - tmin) / size) for (tmin, tmax, size) in dim_params]
total_cells = int(np.prod(dim_n))
for g, n_g in zip(dims, dim_n):
    print(f'{g}: {n_g} cells', file=sys.stderr)


def node_to_tfms(node):
    return tuple((g, tmin + lo * size, tmin + hi * size)
                 for g, (tmin, tmax, size), (lo, hi) in zip(dims, dim_params, node))


def node_cells(node):
    return int(np.prod([hi - lo for (lo, hi) in node]))


def split_node(node):
    widths = [hi - lo for (lo, hi) in node]
    d = max(range(len(widths)), key=lambda k: widths[k])
    if widths[d] <= max(1, args.min_cells):
        return None  # leaf (single cell, or --min-cells wide)
    lo, hi = node[d]
    mid = lo + (hi - lo) // 2
    left, right = list(node), list(node)
    left[d] = (lo, mid)
    right[d] = (mid, hi)
    c = dim_n[d] / 2
    if abs((mid + hi) / 2 - c) > abs((lo + mid) / 2 - c):
        return tuple(right), tuple(left)
    return tuple(left), tuple(right)


bs_state = {'bs': eval_batch_size}   # current batch size; halves on CUDA out-of-memory
prof = {'grid': 0.0, 'bounds': 0.0}


def _sync():
    if args.profile and device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_batch_multi(batch_lower, batch_upper, labels):
    ptb = PerturbationLpNorm(norm=np.inf, x_L=batch_lower, x_U=batch_upper)
    bounded_inputs = BoundedTensor(batch_lower, ptb)

    if cert_method in ('CROWN-IBP', 'CROWN'):
        C = make_spec_matrix(labels, num_classes)
        lb, ub = ibp_net.compute_bounds(x=(bounded_inputs,), method=cert_method, C=C)
        return (lb > 0).all(dim=1)
    elif cert_method == 'IBP-C':
        C = make_spec_matrix(labels, num_classes)
        lb, ub = ibp_net.compute_bounds(x=(bounded_inputs,), method='IBP', C=C)
        return (lb > 0).all(dim=1)
    else:
        lb, ub = ibp_net.compute_bounds(x=(bounded_inputs,), method='IBP')
        labels_onehot = F.one_hot(labels, num_classes=num_classes)
        interval_outputs = lb * labels_onehot + ub * torch.logical_not(labels_onehot)
        _, predicted = torch.max(interval_outputs.data, 1)
        return predicted == labels


init_cells = args.init_cells if args.init_cells > 0 else max(dim_n)
chunks = [[(s, min(s + init_cells, n_g)) for s in range(0, n_g, init_cells)] for n_g in dim_n]
roots = list(itertools.product(*chunks))
roots.sort(key=lambda r: -sum(abs((lo + hi) / 2 - n / 2) for (lo, hi), n in zip(r, dim_n)))  # extremes first
print(f'Starting from {len(roots)} box(es), {total_cells} cells in total', file=sys.stderr)

alive = torch.ones(n_img, dtype=torch.bool, device=device)               
resolved = torch.zeros(n_img, dtype=torch.long, device=device)           
evals = torch.zeros(n_img, dtype=torch.long, device=device)              
stack = []                                                                
root_i = 0
nodes_evaluated = 0
elapsed_prev = 0.0
sig = (n_img, total_cells, init_cells, cert_method, num_correct, args.min_cells)
ckpt_path = shard_ckpt_path(shard_id, num_shards)

if args.resume and os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    if ck['sig'] == sig:
        alive = ck['alive'].to(device)
        resolved = ck['resolved'].to(device)
        evals = ck['evals'].to(device) if 'evals' in ck else evals
        stack = [(node, imgs.to(device)) for node, imgs in ck['stack']]
        root_i, nodes_evaluated, elapsed_prev = ck['root_i'], ck['nodes_evaluated'], ck['elapsed']
        print(f'[shard {shard_id}] resumed from checkpoint ({nodes_evaluated} nodes, {elapsed_prev:.0f}s done)', file=sys.stderr)
    else:
        print(f'[shard {shard_id}] checkpoint does not match this run, starting fresh', file=sys.stderr)


def save_checkpoint():
    tmp = ckpt_path + '.tmp'
    torch.save({'sig': sig, 'alive': alive.cpu(), 'resolved': resolved.cpu(), 'evals': evals.cpu(),
                'stack': [(node, imgs.cpu()) for node, imgs in stack],
                'root_i': root_i, 'nodes_evaluated': nodes_evaluated,
                'elapsed': elapsed_prev + (timer() - s_time)}, tmp)
    os.replace(tmp, ckpt_path)


print(f'Certifying using {cert_method} (eval_batch_size={eval_batch_size})...', file=sys.stderr)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device=device)
s_time = timer()
last_ckpt = timer()

@torch.no_grad()
def falsify(n_points):
    d = len(dims)
    lows = torch.tensor([float(p[0]) for p in dim_params])
    highs = torch.tensor([float(p[1]) for p in dim_params])
    gen = torch.Generator().manual_seed(args.seed)
    if d == 1:
        t = torch.linspace(0, 1, n_points).unsqueeze(1)
    else:
        t = torch.rand(n_points, d, generator=gen)
        corners = torch.tensor(list(itertools.product([0., 1.], repeat=d))) if d <= 4 else torch.zeros(0, d)
        k = min(len(corners), n_points)
        t[:k] = corners[:k]
    vals = lows + t * (highs - lows)
    fb = 4 * eval_batch_size
    for v in vals.tolist():
        idx_all = alive.nonzero().flatten()
        if idx_all.numel() == 0:
            break
        tfms = [(g, x, x) for g, x in zip(dims, v)]
        interp_t = [q for q in tfms if q[0] in geometric_perturbations]
        pix_t = [q for q in tfms if q[0] in pixelwise_perturbations]
        grid = make_interp_grid(img_shape, interp_t) if has_geometric else None
        for s in range(0, idx_all.numel(), fb):
            idx = idx_all[s:s + fb]
            xin = fetch_inputs(idx)
            iv = IntervalTensor(xin, xin)
            if has_geometric:
                iv = spatial(iv, grid)
            if has_pixelwise:
                iv = pixelwise(iv, pix_t)
            pred = net(0.5 * (iv.lower + iv.upper)).argmax(1)
            wrong = idx[pred != cy[idx]]
            if wrong.numel() > 0:
                resolved[wrong] = total_cells
                alive[wrong] = False


if args.falsify_points > 0 and nodes_evaluated == 0 and root_i == 0:
    n_before = int(alive.sum().item())
    falsify(args.falsify_points)
    print(f'[shard {shard_id}] falsification ({args.falsify_points} points): '
          f'{n_before - int(alive.sum().item())} / {n_before} images rejected up front', file=sys.stderr)

pbar = tqdm(total=n_img * total_cells, unit='cell', unit_scale=True, mininterval=10,
            desc=f'GPU{shard_id}', position=shard_id, file=sys.stderr)
pbar.update(int(resolved.sum().item()))

def reject(idx):
    pbar.update(int(total_cells * idx.numel() - resolved[idx].sum().item()))
    resolved[idx] = total_cells
    alive[idx] = False

timed_out = False
with torch.no_grad():
    while True:
        if args.time_limit > 0 and elapsed_prev + (timer() - s_time) > args.time_limit:
            timed_out = True
            break
            
        if not stack:
            if root_i >= len(roots):
                break
            imgs0 = alive.nonzero().flatten()
            if imgs0.numel() == 0:
                break
            stack.append((roots[root_i], imgs0))
            root_i += 1
            continue

        batch_nodes = []
        batch_images = []
        batch_lower_list = []
        batch_upper_list = []
        batch_labels_list = []
        
        current_batch_size = 0
        
        while stack and current_batch_size < bs_state['bs']:
            node, imgs = stack.pop()
            imgs = imgs[alive[imgs]]  
            if imgs.numel() == 0:
                continue
                
            num_imgs = imgs.numel()
            
            if current_batch_size + num_imgs > bs_state['bs'] and current_batch_size > 0:
                stack.append((node, imgs))
                break

            interp_transforms = []
            pixelwise_transforms = []
            for (p_name, pmin, pmax) in node_to_tfms(node):
                if p_name in geometric_perturbations:
                    interp_transforms.append((p_name, pmin, pmax))
                elif p_name in pixelwise_perturbations:
                    pixelwise_transforms.append((p_name, pmin, pmax))

            _sync(); t0 = timer()
            interp_grid = make_interp_grid(img_shape, interp_transforms) if has_geometric else None
            _sync(); prof['grid'] += timer() - t0

            inputs = fetch_inputs(imgs)
            labels = cy[imgs]
            interval_inputs = IntervalTensor(inputs, inputs)
            
            if has_geometric:
                interval_inputs = spatial(interval_inputs, interp_grid)
            if has_pixelwise:
                interval_inputs = pixelwise(interval_inputs, pixelwise_transforms)

            batch_lower_list.append(interval_inputs.lower)
            batch_upper_list.append(interval_inputs.upper)
            batch_labels_list.append(labels)
            
            batch_nodes.append((node, imgs))
            current_batch_size += num_imgs
            nodes_evaluated += 1

        if current_batch_size == 0:
            continue

        combined_l = torch.cat(batch_lower_list, dim=0)
        combined_u = torch.cat(batch_upper_list, dim=0)
        combined_labels = torch.cat(batch_labels_list, dim=0)

        _sync(); t0 = timer()
        oom = False
        try:
            batch_ok = run_batch_multi(combined_l, combined_u, combined_labels)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or bs_state['bs'] == 1:
                raise
            oom = True
            
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            bs_state['bs'] = max(1, bs_state['bs'] // 2)
            print(f'[shard {shard_id}] CUDA OOM -> batch size now {bs_state["bs"]}', file=sys.stderr)
            for node, imgs in reversed(batch_nodes):
                stack.append((node, imgs))
                nodes_evaluated -= 1 
            continue
            
        _sync(); prof['bounds'] += timer() - t0

        offset = 0
        for node, imgs in batch_nodes:
            num_imgs = imgs.numel()
            ok = batch_ok[offset : offset + num_imgs]
            offset += num_imgs
            
            evals[imgs] += 1
            good, bad = imgs[ok], imgs[~ok]

            if good.numel() > 0:
                c = node_cells(node)
                resolved[good] += c
                pbar.update(c * good.numel())
            if bad.numel() > 0:
                children = split_node(node)
                if children is None: 
                    reject(bad)
                else:
                    if args.max_node_evals > 0: 
                        over = evals[bad] > args.max_node_evals
                        if over.any():
                            reject(bad[over])
                            bad = bad[~over]
                    if bad.numel() > 0:
                        for ch in reversed(children):  
                            stack.append((ch, bad))

        if nodes_evaluated % 20 == 0:
            pbar.set_postfix(alive=int(alive.sum().item()), nodes=nodes_evaluated, bs=bs_state['bs'], refresh=False)
        if timer() - last_ckpt > args.ckpt_interval:
            save_checkpoint()
            last_ckpt = timer()

pbar.close()
if timed_out:
    alive &= resolved >= total_cells
    print(f'[shard {shard_id}] TIME LIMIT hit after {args.time_limit:.0f}s: unfinished images counted as not certified', file=sys.stderr)

if device.type == "cuda":
    torch.cuda.synchronize(device)
certify_time = elapsed_prev + (timer() - s_time)

robust_indices = shard_global[alive.nonzero().flatten().cpu()].tolist()
max_gpu_memory_bytes = torch.cuda.max_memory_allocated(device=device) if device.type == "cuda" else 0

print(f'[shard {shard_id}] nodes evaluated: {nodes_evaluated}, certified {len(robust_indices)} / {n_img}, time {certify_time:.1f}s', file=sys.stderr)
if args.profile:
    print(f'[shard {shard_id}] time in make_interp_grid: {prof["grid"]:.1f}s, in spatial+bounds: {prof["bounds"]:.1f}s', file=sys.stderr)

torch.save({'robust_indices': robust_indices, 'num_correct': num_correct, 'dataset_len': dataset_len,
            'cert_time': certify_time, 'max_mem': max_gpu_memory_bytes, 'nodes': nodes_evaluated},
           shard_result_path(shard_id, num_shards))
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

if standalone:
    report_and_save(robust_indices, num_correct, dataset_len, certify_time, max_gpu_memory_bytes)