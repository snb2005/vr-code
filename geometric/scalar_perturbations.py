import torch

def contrast_brightness(img, c, b):
    return torch.clamp(c * img + b, min=0.0, max=1.0)
