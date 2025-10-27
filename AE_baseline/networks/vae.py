"""
Variational Autoencoder (VAE) for CQ500 Anomaly Detection

VAE extends the standard AE with a probabilistic bottleneck layer
that learns a distribution over the latent space instead of a
deterministic mapping.
"""

from networks.ae import AE
from networks.base_units.blocks import VaeBottleNeck


class VAE(AE):
    """
    Variational Autoencoder
    
    Architecture:
        - Same encoder/decoder structure as AE
        - Bottleneck: Variational bottleneck with reparameterization trick
    
    Args:
        input_size: Input image size (default: 64)
        in_planes: Input channels (default: 1 for grayscale)
        base_width: Base channel width (default: 16)
        expansion: Channel expansion factor (default: 1)
        mid_num: Hidden size in bottleneck (default: 2048)
        latent_size: Latent vector size (default: 16)
        en_num_layers: Encoder block depth (default: 1)
        de_num_layers: Decoder block depth (default: 1)
    
    Forward returns:
        dict with:
            - x_hat: Reconstructed image
            - mu: Mean of latent distribution
            - log_var: Log variance of latent distribution
            - en_features: Encoder feature maps
            - de_features: Decoder feature maps
    """
    
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1):
        super(VAE, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size, en_num_layers,
                                  de_num_layers)
        # Replace standard bottleneck with VAE bottleneck
        self.bottle_neck = VaeBottleNeck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                         latent_size=latent_size)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, 1, H, W)
        
        Returns:
            dict with x_hat, mu, log_var, and feature maps
        """
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        de4, mu, log_var = bottle_out['out'], bottle_out['mu'], bottle_out["log_var"]

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        return {'x_hat': x_hat, 'log_var': log_var, 'mu': mu,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}

