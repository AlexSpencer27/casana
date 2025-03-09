import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import MultiScaleCNNBranch, SkipConnectionMLP, PeakOrderingLayer, AdaptiveFeaturePooling

@register_model("multi_scale_conv1d")
class MultiScaleConv1D(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Multi-scale convolution using component
        self.multi_scale_branch = MultiScaleCNNBranch(
            in_channels=1,
            channels_per_kernel=16,
            kernel_sizes=(7, 15, 31),
            pooling='max'
        )
        
        # Second convolution layer after concatenation
        self.conv2 = nn.Conv1d(48, 64, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool1d(2)
        
        # Third convolution layer
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        
        # Adaptive pooling to fix output size
        self.adaptive_pool = AdaptiveFeaturePooling(output_size=16, pooling_type='avg')
        
        # Fully connected layers with skip connection using component
        self.fc_block = SkipConnectionMLP(
            input_dim=64 * 16,
            hidden_dim=256,
            output_dim=64,
            dropout_rate=0.3
        )
        
        # Output layer
        self.output = nn.Linear(64, 3)
        
        # Peak ordering layer
        self.peak_ordering = PeakOrderingLayer(softness=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Multi-scale time domain processing using component
        x = self.multi_scale_branch(x)
        
        # Second convolution block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Third convolution block
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Flatten features
        x = x.view(batch_size, -1)
        
        # Process through FC layers with skip connection
        x = self.fc_block(x)
        
        # Output layer
        x = self.output(x)
        
        # Ensure peak1 < midpoint < peak2 using component
        x = self.peak_ordering(x)
        
        return x 