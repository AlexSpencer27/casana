"""
Common utilities and components used across models.
"""

import torch
import torch.nn as nn

class PeakOrderingLayer(nn.Module):
    def __init__(self):
        """
        Simple layer to ensure predictions are in valid range and compute midpoint.
        """
        super().__init__()
        
    def forward(self, x):
        """
        Apply basic constraints to ensure valid peak positions.
        
        Args:
            x: Input tensor of shape [batch_size, 3] representing normalized positions
               [peak1, midpoint, peak2]
            
        Returns:
            Tensor of shape [batch_size, 3] with constrained positions in [0,1]
        """
        # Hard clamp to [0,1] range
        x = torch.clamp(x, 0.0, 1.0)
        
        # Extract peaks and midpoint
        peak1, _, peak2 = x.split(1, dim=1)
        
        # Compute midpoint as median between peaks
        midpoint = (peak1 + peak2) * 0.5
        
        # Concatenate back to [batch_size, 3]
        x = torch.cat([peak1, midpoint, peak2], dim=1)
        
        return x
    
    
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