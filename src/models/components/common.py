"""
Common utilities and components used across models.
"""

import torch
import torch.nn as nn

class BoundedPeakOutput(nn.Module):
    def __init__(self):
        """
        A better approach for ensuring peak predictions are in the [0,1] range
        using sigmoid activation.
        """
        super().__init__()
        self.peak1_min = 0.05
        self.peak1_max = 0.25
        self.peak2_min = 0.3
        self.peak2_max = 0.9
        
    def forward(self, x):
        """
        Apply sigmoid to get values in [0,1] range and compute proper midpoint.
        
        Args:
            x: Input tensor of shape [batch_size, 3] representing unbounded positions
               [peak1, _, peak2]
            
        Returns:
            Tensor of shape [batch_size, 3] with positions in [0,1]
        """
        return x
        # Get raw values
        raw_peak1, _, raw_peak2 = x.split(1, dim=1)
        
        # Use sigmoid for bounded output in (0,1)
        peak1_range = self.peak1_max - self.peak1_min
        peak2_range = self.peak2_max - self.peak2_min
        
        # Use sigmoid for more stable bounds
        peak1 = self.peak1_min + peak1_range * torch.sigmoid(raw_peak1)
        peak2 = self.peak2_min + peak2_range * torch.sigmoid(raw_peak2)
        
        # Compute midpoint
        midpoint = (peak1 + peak2) * 0.5
        
        return torch.cat([peak1, midpoint, peak2], dim=1)


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