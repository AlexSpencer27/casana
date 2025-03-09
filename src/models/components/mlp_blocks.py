"""
MLP (Multi-Layer Perceptron) blocks with various architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3, use_batchnorm=True):
        """
        Fully connected layers with residual connections.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout_rate: Dropout probability (default: 0.3)
            use_batchnorm: Whether to use batch normalization (default: True)
        """
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        # Main path
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Skip connection
        self.fc_skip = nn.Linear(input_dim, output_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        """
        Forward pass through the MLP with skip connection.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        # Main path
        if self.use_batchnorm:
            main = F.relu(self.bn1(self.fc1(x)))
        else:
            main = F.relu(self.fc1(x))
            
        main = self.dropout(main)
        
        if self.use_batchnorm:
            main = F.relu(self.bn2(self.fc2(main)))
        else:
            main = F.relu(self.fc2(main))
        
        # Skip connection
        skip = F.relu(self.fc_skip(x))
        
        # Combine
        x = main + skip
        return self.dropout(x)


class ProgressiveMLP(nn.Module):
    def __init__(self, layer_dims, dropout_rate=0.3, use_batchnorm=True):
        """
        Progressive MLP with multiple layers and optional batch normalization.
        
        Args:
            layer_dims: List of layer dimensions, including input and output dimensions
            dropout_rate: Dropout probability (default: 0.3)
            use_batchnorm: Whether to use batch normalization (default: True)
        """
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        # Create layers
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batchnorm else None
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if use_batchnorm and i < len(layer_dims) - 2:  # No BN for output layer
                self.bn_layers.append(nn.BatchNorm1d(layer_dims[i+1]))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass through the progressive MLP.
        
        Args:
            x: Input tensor of shape [batch_size, layer_dims[0]]
            
        Returns:
            Tensor of shape [batch_size, layer_dims[-1]]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply batch norm and activation for all but the last layer
            if i < len(self.layers) - 1:
                if self.use_batchnorm:
                    x = self.bn_layers[i](x)
                x = F.relu(x)
                x = self.dropout(x)
                
        return x 