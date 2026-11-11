import torch
import torch.nn as nn

class NormalizeInput(nn.Module):
    def __init__(self, mean, std, channels=3) :
        super(NormalizeInput, self).__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.channels = channels
        
    def forward(self, input):
        mean = self.mean.reshape(1, self.channels, 1, 1)
        std = self.std.reshape(1, self.channels, 1, 1)
        return (input - mean) / std
