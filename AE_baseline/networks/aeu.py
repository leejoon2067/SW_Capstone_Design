"""
Autoencoder with Uncertainty (AE-U) for CQ500 Anomaly Detection
Predicts reconstruction uncertainty (log variance)
"""

from networks.ae import AE
from networks.base_units.blocks import BasicBlock


class AEU(AE):
    """
    Autoencoder with Uncertainty estimation
    
    Extends AE by predicting log variance in addition to reconstruction.
    
    Output:
        - x_hat: Reconstructed image
        - log_var: Log variance (uncertainty estimation)
        - z: Latent representation
    
    TODO [향후 개선]:
        - Aleatoric vs Epistemic uncertainty 분리
        - Uncertainty calibration
    """
    
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1):
        super(AEU, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size, en_num_layers,
                                  de_num_layers)

        # Last decoder block outputs 2*in_planes (reconstruction + log_var)
        self.de_block4 = BasicBlock(1 * base_width * expansion, 2 * in_planes, de_num_layers, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        """
        Forward pass with uncertainty
        
        Args:
            x: Input image (B, 1, H, W)
        
        Returns:
            dict with:
                - x_hat: Reconstructed image (B, 1, H, W)
                - log_var: Log variance (B, 1, H, W)
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
        
        # Split output into reconstruction and log variance
        x_hat, log_var = self.de_block4(de1).chunk(2, 1)

        return {'x_hat': x_hat, 'log_var': log_var, 'z': z,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}