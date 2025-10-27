"""
Weight Normalized Convolution Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WNConv2d(nn.Module):
    """
    Weight Normalized 2D Convolution
    
    Applies weight normalization to standard Conv2d layer.
    Weight normalization helps with training stability and convergence.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)

