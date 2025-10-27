"""
Loss Functions for AE/AE-U models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AELoss(nn.Module):
    """
    Basic Autoencoder Loss (MSE)
    
    Args:
        grad_score: Use gradient-based anomaly score (default: False)
    """
    
    def __init__(self, grad_score=False):
        super(AELoss, self).__init__()
        self.grad_score = grad_score

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        """
        Args:
            net_in: Input image (B, C, H, W)
            net_out: Network output dict with 'x_hat'
            anomaly_score: Return anomaly score instead of loss
            keepdim: Keep spatial dimensions for anomaly map
        
        Returns:
            loss (scalar) or anomaly_score (B,) or anomaly_map (B, 1, H, W)
        """
        x_hat = net_out['x_hat']
        loss = (net_in - x_hat) ** 2
    
        if anomaly_score:
            if self.grad_score:
                # Gradient-based score
                grad = torch.abs(torch.autograd.grad(loss.mean(), net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True) if keepdim else torch.mean(grad, dim=[1, 2, 3])
            else:
                # Reconstruction error score
                return torch.mean(loss, dim=[1], keepdim=True) if keepdim else torch.mean(loss, dim=[1, 2, 3])
        else:
            return loss.mean()


class L1Loss(nn.Module):
    """L1 (MAE) Loss for AE"""
    
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        loss = torch.abs(net_in - x_hat)

        if anomaly_score:
            return torch.mean(loss, dim=[1], keepdim=True) if keepdim else torch.mean(loss, dim=[1, 2, 3])
        else:
            return loss.mean()


class AEULoss(nn.Module):
    """
    Autoencoder with Uncertainty Loss
    
    Loss = exp(-log_var) * reconstruction_loss + log_var
    
    This encourages the model to predict uncertainty (log_var) 
    where reconstruction is difficult.
    """
    
    def __init__(self):
        super(AEULoss, self).__init__()

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        """
        Args:
            net_in: Input image
            net_out: Dict with 'x_hat' and 'log_var'
            
        Returns:
            If training: (total_loss, recon_loss, log_var_mean)
            If testing: anomaly_score based on uncertainty-weighted reconstruction
        """
        x_hat, log_var = net_out['x_hat'], net_out['log_var']
        recon_loss = (net_in - x_hat) ** 2

        # Uncertainty-weighted reconstruction loss
        loss1 = torch.exp(-log_var) * recon_loss
        loss = loss1 + log_var

        if anomaly_score:
            # Use uncertainty-weighted reconstruction for anomaly detection
            return torch.mean(loss1, dim=[1], keepdim=True) if keepdim else torch.mean(loss1, dim=[1, 2, 3])
        else:
            # Return total loss + individual components for logging
            return loss.mean(), recon_loss.mean().item(), log_var.mean().item()


class SSIMLoss(nn.Module):
    """
    SSIM-based Loss
    
    TODO [향후 개선]: 
        - SSIM implementation 필요
        - Medical image에 최적화된 similarity metric 고려
    """
    
    def __init__(self, win_size=11):
        super(SSIMLoss, self).__init__()
        self.win_size = win_size

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        raise NotImplementedError("SSIM loss needs to be implemented")


class CombinedLoss(nn.Module):
    """
    Combined loss (MSE + L1 + etc.)
    
    TODO [향후 개선]:
        - Multiple loss combination
        - Weighted loss scheduling
    """
    
    def __init__(self, mse_weight=1.0, l1_weight=0.0):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
    
    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        
        mse_loss = (net_in - x_hat) ** 2
        l1_loss = torch.abs(net_in - x_hat)
        
        loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        if anomaly_score:
            return torch.mean(loss, dim=[1], keepdim=True) if keepdim else torch.mean(loss, dim=[1, 2, 3])
        else:
            return loss.mean()


class VAELoss(nn.Module):
    """
    Variational Autoencoder Loss
    
    Loss = Reconstruction Loss + KL_weight * KL Divergence
    
    Where:
        - Reconstruction Loss: MSE between input and output
        - KL Divergence: KL(q(z|x) || p(z)) where p(z) ~ N(0, I)
    
    Args:
        kl_weight: Weight for KL divergence term (default: 0.005)
        grad: Gradient-based anomaly score type (default: None)
              Options: 'elbo', 'rec', 'kl', 'combi'
    """
    
    def __init__(self, kl_weight=0.005, grad=None):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        self.grad = grad

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        """
        Args:
            net_in: Input image
            net_out: Dict with 'x_hat', 'mu', and 'log_var'
            anomaly_score: Return anomaly score instead of loss
            keepdim: Keep spatial dimensions
        
        Returns:
            If training: (total_loss, recon_loss, kl_loss)
            If testing: anomaly_score
        """
        x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
        
        # Reconstruction loss (MSE)
        recon_loss = (net_in - x_hat) ** 2
        
        # KL divergence: -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
        
        # Total loss
        loss = recon_loss.mean() + self.kl_weight * kl_loss.mean()

        if anomaly_score:
            # Different gradient-based anomaly scores
            if self.grad == 'elbo':
                grad = torch.abs(torch.autograd.grad(loss, net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True) if keepdim else torch.mean(grad, dim=[1, 2, 3])
            elif self.grad == 'rec':
                grad = torch.abs(torch.autograd.grad(recon_loss.mean(), net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True) if keepdim else torch.mean(grad, dim=[1, 2, 3])
            elif self.grad == 'kl':
                grad = torch.abs(torch.autograd.grad(kl_loss.mean(), net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True) if keepdim else torch.mean(grad, dim=[1, 2, 3])
            elif self.grad == 'combi':
                kl_grad = torch.abs(torch.autograd.grad(kl_loss.mean(), net_in)[0])
                combi = recon_loss * kl_grad
                return torch.mean(combi, dim=[1], keepdim=True) if keepdim else torch.mean(combi, dim=[1, 2, 3])
            else:
                # Default: use reconstruction error as anomaly score
                return torch.mean(recon_loss, dim=[1], keepdim=True) if keepdim \
                    else torch.mean(recon_loss, dim=[1, 2, 3])
        else:
            # Return total loss + individual components for logging
            return loss, recon_loss.mean().item(), kl_loss.mean().item()