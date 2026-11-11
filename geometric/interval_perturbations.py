import torch
import torch.nn.functional as F
import numpy as np

from .intervals import *

import sys
sys.path.append('../')
from definitions import *

geometric_perturbations = [ROTATE, TRANSLATE_X, TRANSLATE_Y, SCALE, SHEAR]
pixelwise_perturbations = [CONTRAST, BRIGHTNESS]

def pixelwise(img, transforms):
    for tfm_params in transforms:
        transform, theta_min, theta_max = tfm_params
        if transform == CONTRAST:
            img = IntervalTensor((img.lower * theta_min).clamp(0, 1), (img.upper * theta_max).clamp(0, 1))
        elif transform == BRIGHTNESS:
            img = IntervalTensor((img.lower + theta_min).clamp(0, 1), (img.upper + theta_max).clamp(0, 1))
    return img

# transforms is a list with tuples of the form (transform_name, theta_min, theta_max)
# make sure to specify transforms in the same order as they should be applied
def make_interp_grid(img_shape, transforms, device='cuda'):
    _, img_h, img_w = img_shape

    # row/col to x/y
    x_indices = torch.tensor([k for k in range(img_w)], device=device)
    y_indices = torch.tensor([k for k in range(img_h)], device=device)
    half_img_width = (img_w - 1) / 2.0
    half_img_height = (img_h - 1) / 2.0
    x = x_indices - half_img_width
    y = half_img_height - y_indices
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

    # calculate inverse coordinates
    inv_coords_x = IntervalTensor(x_grid, x_grid)
    inv_coords_y = IntervalTensor(y_grid, y_grid)

    for tfm_params in reversed(transforms):
        transform, theta_min, theta_max = tfm_params
        
        # convert degrees to radians
        if transform == ROTATE:
            theta_min = theta_min * np.pi / 180
            theta_max = theta_max * np.pi / 180

        theta = IntervalTensor(torch.tensor([theta_min], device=device), torch.tensor([theta_max], device=device))

        if transform == ROTATE:
            tmp_x = inv_coords_x
            tmp_y = inv_coords_y
            inv_coords_x = tmp_x * cos_i(theta) + tmp_y * sin_i(theta)
            inv_coords_y = -tmp_x * sin_i(theta) + tmp_y * cos_i(theta)
        elif transform == TRANSLATE_X:
            inv_coords_x = inv_coords_x - theta
        elif transform == TRANSLATE_Y:
            inv_coords_y = inv_coords_y - theta
        elif transform == SCALE:
            inv_coords_x = inv_coords_x / theta
            inv_coords_y = inv_coords_y / theta
        elif transform == SHEAR:
            inv_coords_x = inv_coords_x - inv_coords_y * theta

    # interpolate
    x_distances = relu_i(1 - abs_i(inv_coords_x.reshape((-1, 1)) - IntervalTensor(x, x)))
    y_distances = relu_i(1 - abs_i(inv_coords_y.reshape((-1, 1)) - IntervalTensor(y, y)))
    interp_grid_l = torch.einsum('py, px->pyx', y_distances.lower, x_distances.lower)
    interp_grid_u = torch.einsum('py, px->pyx', y_distances.upper, x_distances.upper)
    interp_grid = IntervalTensor(interp_grid_l, interp_grid_u)

    # convert to sparse format
    nonzero_mask = (interp_grid.upper != 0)
    nonzero_counts = nonzero_mask.sum((-2, -1))
    
    interp_grid_flatten = interp_grid.flatten()
    nonzero_vals = interp_grid_flatten[nonzero_mask.flatten()]         # stores the actual values for all nonzero interpolation entries
    nonzero_indices = torch.nonzero(nonzero_mask.flatten()).flatten()  # stores index of each nonzero interpolation entry in flattened grid
    nonzero_rc_indices = nonzero_indices % (img_h * img_w)
    nonzero_row_indices = torch.div(nonzero_rc_indices, img_w, rounding_mode='trunc')  # for each nonzero interpolation entry, stores its x coordinate in image
    nonzero_col_indices = nonzero_rc_indices % img_w                                   # for each nonzero interpolation entry, stores its y coordinate in image

    # nonzero_counts will be vector of length img_h*img_w (one count for each pixel)
    # nonzero_row/col_indices and nonzero_vals will be vectors of length: # of nonzero entries in interp_grid
    return (nonzero_row_indices, nonzero_col_indices, nonzero_vals, nonzero_counts)

def spatial(img, interp_grid, device='cuda'):
    (nonzero_row_indices, nonzero_col_indices, nonzero_vals, nonzero_counts) = interp_grid

    # multiply all nonzero interpolation entries with the image pixels in their corresponding coordinates; we now obtain the weighted pixel values
    interp_mul = mul_pos(nonzero_vals, img[:, :, nonzero_row_indices, nonzero_col_indices])

    # sum the interpolation entries that belong to the same pixel and reshape back to image dimension
    # https://stackoverflow.com/questions/55567838/how-to-avoid-split-and-sum-of-pieces-in-pytorch-or-numpy
    ind = torch.arange(len(nonzero_counts), device=device).repeat_interleave(nonzero_counts)
    img_l = torch.zeros((img.shape[0], img.shape[1], len(nonzero_counts)), device=device)
    img_l.index_add_(2, ind, interp_mul.lower)
    img_u = torch.zeros((img.shape[0], img.shape[1], len(nonzero_counts)), device=device)
    img_u.index_add_(2, ind, interp_mul.upper)
    img = IntervalTensor(img_l.view(*img.shape).clamp(0, 1), img_u.view(*img.shape).clamp(0, 1))
    
    return img
