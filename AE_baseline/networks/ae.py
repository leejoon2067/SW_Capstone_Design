"""
Autoencoder (AE) for CQ500 Anomaly Detection
2D CNN-based architecture
"""

import torch.nn as nn
from networks.base_units.blocks import BasicBlock, BottleNeck, SpatialBottleNeck


class AE(nn.Module):
    """
    Autoencoder for 2D medical image reconstruction
    
    Architecture:
        Encoder: 4 downsampling blocks (input_size → input_size/16)
        Bottleneck: Linear or Spatial bottleneck
        Decoder: 4 upsampling blocks (input_size/16 → input_size)
    
    Args:
        input_size: Input image size (default: 64)
        in_planes: Input channels (default: 1 for grayscale)
        base_width: Base channel width (default: 16)
        expansion: Channel expansion factor (default: 1)
        mid_num: Hidden size in bottleneck (default: 2048)
        latent_size: Latent vector size (default: 16)
        en_num_layers: Encoder block depth (default: 1)
        de_num_layers: Decoder block depth (default: 1)
        spatial: Use spatial bottleneck instead of linear (default: False)
    
    TODO [향후 개선]:
        - Multi-scale features
        - Attention mechanism
        - Skip connections (U-Net style)
    """
    
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1, spatial=False):
        super(AE, self).__init__()

        bottleneck = SpatialBottleNeck if spatial else BottleNeck

        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16

        # Encoder blocks
        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)
        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)

        # Bottleneck
        self.bottle_neck = bottleneck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                      latent_size=latent_size)

        # Decoder blocks
        self.de_block1 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block2 = BasicBlock(4 * base_width * expansion, 2 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block3 = BasicBlock(2 * base_width * expansion, 1 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block4 = BasicBlock(1 * base_width * expansion, in_planes, de_num_layers, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, 1, H, W)
        
        Returns:
            dict with:
                - x_hat: Reconstructed image (B, 1, H, W)
                - z: Latent representation
                - en_features: Encoder feature maps
                - de_features: Decoder feature maps
        """
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de4 = bottle_out['z'], bottle_out['out']

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        return {'x_hat': x_hat, 'z': z, 'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}