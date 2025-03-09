"""
Multi-scale CNN components for processing signals with different kernel sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNNBranch(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        channels_per_kernel=16,
        kernel_sizes=(7, 15, 31), 
        pooling='max',
        pooling_size=2,
        activation=F.relu
    ):
        """
        Multi-scale CNN branch that processes input with parallel convolutions using different kernel sizes.
        
        Args:
            in_channels: Number of input channels (default: 1)
            channels_per_kernel: Number of output channels per kernel (default: 16)
            kernel_sizes: Tuple of kernel sizes (default: (7, 15, 31))
            pooling: Type of pooling to apply ('max', 'avg', or None) (default: 'max')
            pooling_size: Size of pooling window (default: 2)
            activation: Activation function to use (default: F.relu)
        """
        super().__init__()
        
        self.activation = activation
        
        # Create a conv layer for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, channels_per_kernel, k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Define pooling layer based on type
        self.pool = self._get_pooling(pooling, pooling_size)
        
        # Total output channels after concatenation
        self.out_channels = channels_per_kernel * len(kernel_sizes)
        
    def _get_pooling(self, pooling_type, size):
        """Get pooling layer based on type."""
        if pooling_type == 'max':
            return nn.MaxPool1d(size)
        elif pooling_type == 'avg':
            return nn.AvgPool1d(size)
        else:
            return nn.Identity()
    
    def forward(self, x):
        """
        Forward pass for the multi-scale CNN branch.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, signal_length]
            
        Returns:
            Tensor of shape [batch_size, out_channels, signal_length/pooling_size]
        """
        # Apply each convolution and activation
        conv_outputs = [self.activation(conv(x)) for conv in self.convs]
        
        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)
        
        # Apply pooling if defined
        if self.pool is not None:
            x = self.pool(x)
            
        return x 