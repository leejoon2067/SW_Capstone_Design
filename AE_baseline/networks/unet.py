"""
U-Net for CQ500 Anomaly Detection

U-Net is a fully convolutional network with skip connections
that preserves spatial information better than standard autoencoders.
"""

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from networks.base_units.swish import CustomSwish
from networks.base_units.ws_conv import WNConv2d


def get_groups(channels: int) -> int:
    """
    Calculate suitable number of groups for GroupNormalization
    
    Args:
        channels: Number of channels
    
    Returns:
        Optimal number of groups (median divisor)
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]


class UNet(nn.Module):
    """
    U-Net Architecture
    
    A modified U-Net implementation based on:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        n_classes: Number of output channels (default: 1 for reconstruction)
        depth: Network depth (default: 5)
        wf: Width factor - first layer has 2**wf filters (default: 6 -> 64 filters)
        padding: If True, apply padding to maintain input shape (default: True)
        norm: Normalization type - 'batch' or 'group' (default: 'group')
        up_mode: Upsampling mode - 'upconv' or 'upsample' (default: 'upconv')
    """
    
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            depth=5,
            wf=6,
            padding=True,
            norm="group",
            up_mode='upconv'):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        
        # Encoder (downsampling path)
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        # Decoder (upsampling path)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        # Final 1x1 convolution to produce output
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward_down(self, x):
        """Encoder forward pass"""
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)
        return x, blocks

    def forward_up_without_last(self, x, blocks):
        """Decoder forward pass without final layer"""
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)
        return x

    def forward_without_last(self, x):
        """Full forward pass without final layer"""
        x, blocks = self.forward_down(x)
        x = self.forward_up_without_last(x, blocks)
        return x

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            dict with x_hat (reconstructed image)
        """
        x = self.get_features(x)
        return {'x_hat': self.last(x)}

    def get_features(self, x):
        """Get features before final layer (useful for analysis)"""
        return self.forward_without_last(x)


class UNetConvBlock(nn.Module):
    """
    U-Net Convolutional Block
    
    Two consecutive convolutions with normalization and activation
    """
    
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        
        # First convolution
        if padding:
            block.append(nn.ReflectionPad2d(1))
        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())
        
        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        # Second convolution
        if padding:
            block.append(nn.ReflectionPad2d(1))
        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())
        
        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    """
    U-Net Upsampling Block
    
    Upsamples the input and concatenates with skip connection
    """
    
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock, self).__init__()
        
        # Upsampling layer
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        """Center crop layer to match target size"""
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        """
        Forward pass
        
        Args:
            x: Input from previous layer
            bridge: Skip connection from encoder
        
        Returns:
            Upsampled and processed features
        """
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out

