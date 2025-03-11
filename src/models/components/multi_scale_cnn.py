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
        activation=F.relu,
        dropout_rate=0.3
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
            dropout_rate: Dropout rate for regularization (default: 0.3)
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
        
        # Second convolution layer after concatenation
        self.conv2 = nn.Conv1d(self.out_channels, 64, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool1d(2)
        
        # Third convolution layer
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        
        # Adaptive pooling to fix output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        # Fully connected layers with skip connection
        self.fc1 = nn.Linear(64 * 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_skip = nn.Linear(64 * 16, 64)  # Skip connection
        self.output = nn.Linear(64, 3)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        
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
            Tensor of shape [batch_size, 3] with sigmoid-activated outputs
        """
        batch_size = x.size(0)
        
        # Apply each convolution and activation
        conv_outputs = [self.activation(conv(x)) for conv in self.convs]
        
        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)
        
        # Apply first pooling if defined
        if self.pool is not None:
            x = self.pool(x)
            
        # Second convolution block
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        
        # Third convolution block
        x = self.activation(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Flatten features
        x = x.view(batch_size, -1)
        
        # Main path through FC layers
        main = self.activation(self.bn1(self.fc1(x)))
        main = self.dropout(main)
        main = self.activation(self.bn2(self.fc2(main)))
        
        # Skip connection
        skip = self.activation(self.fc_skip(x))
        
        # Combine main path and skip connection
        x = main + skip
        x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        
        # Squash output to 0-1 range
        x = torch.sigmoid(x)
        
        return x 