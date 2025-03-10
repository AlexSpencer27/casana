"""
Common utilities and components used across models.
"""

import torch
import torch.nn as nn

class PeakOrderingLayer(nn.Module):
    def __init__(self, softness=0.1, min_separation=0.1):
        """
        Ensure consistent ordering and valid ranges for peak outputs.
        
        This layer applies constraints to ensure:
        1. All positions are in [0, 1] range (normalized)
        2. peak1 < midpoint < peak2
        3. Maintains minimum separation between peaks
        
        Args:
            softness: Smoothness parameter for differentiable constraints (default: 0.1)
            min_separation: Minimum separation between peaks in normalized space (default: 0.1)
        """
        super().__init__()
        self.softness = softness
        self.min_separation = min_separation
        
    def forward(self, x):
        """
        Apply constraints to ensure valid peak positions.
        
        Args:
            x: Input tensor of shape [batch_size, 3] representing normalized positions
               [peak1, midpoint, peak2]
            
        Returns:
            Tensor of shape [batch_size, 3] with constrained positions in [0,1]
        """
        # Hard clamp to [0,1] range
        x = torch.clamp(x, 0.0, 1.0)
        
        # Extract peaks and midpoint
        peak1, midpoint, peak2 = x.split(1, dim=1)
        
        # Ensure minimum separation between peaks
        min_peak2 = peak1 + self.min_separation
        peak2 = torch.maximum(peak2, min_peak2)
        
        # Recompute midpoint to ensure it's between peaks
        midpoint = peak1 + (peak2 - peak1) * 0.5
        
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