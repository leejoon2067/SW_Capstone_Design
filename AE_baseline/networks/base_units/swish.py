"""
Swish Activation Function
"""

import torch
import torch.nn as nn


class CustomSwish(nn.Module):
    """
    Swish activation function: f(x) = x * sigmoid(x)
    
    More smooth than ReLU and has been shown to work better
    in some deep networks.
    """
    
    def __init__(self):
        super(CustomSwish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

