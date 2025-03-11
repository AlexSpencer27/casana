"""
Common utilities and components used across models.
"""

import torch
import torch.nn as nn

class AdaptiveFeaturePooling(nn.Module):
    def __init__(self, output_size=16, pooling_type='avg'):
        """
        Adaptive pooling for feature maps with flexible input sizes.
        
        Args:
            output_size: Desired output size (default: 16)
            pooling_type: Type of pooling to use ('avg' or 'max') (default: 'avg')
        """
        super().__init__()
        self.output_size = output_size
        
        if pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        elif pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(output_size)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
            
    def forward(self, x):
        """
        Apply adaptive pooling to input feature map.
        
        Args:
            x: Input tensor of shape [batch_size, channels, length]
            
        Returns:
            Tensor of shape [batch_size, channels, output_size]
        """
        return self.pool(x) 