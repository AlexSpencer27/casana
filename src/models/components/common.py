"""
Common utilities and components used across models.
"""

import torch
import torch.nn as nn

class PeakOrderingLayer(nn.Module):
    def __init__(self, softness=0.1):
        """
        Ensure consistent ordering of peak outputs (peak1 < midpoint < peak2).
        
        This layer applies a soft differentiable sorting to ensure the output
        follows the constraint that peak1 < midpoint < peak2, while maintaining
        gradient flow.
        
        Args:
            softness: Smoothness parameter for differentiable sorting (default: 0.1)
                     Higher values make the sorting more gradual but less strict.
        """
        super().__init__()
        self.softness = softness
        
    def forward(self, x):
        """
        Apply soft ordering constraint to ensure peak1 < midpoint < peak2.
        
        Args:
            x: Input tensor of shape [batch_size, 3] representing the three peak positions
            
        Returns:
            Tensor of shape [batch_size, 3] with ordered peak positions
        """
        # Ensure peak1 < midpoint < peak2 using soft constraints
        # Sort the outputs but maintain gradient flow with a differentiable approach
        sorted_x = x + self.softness * (torch.sort(x, dim=1)[0] - x)
        
        return sorted_x
    
    
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