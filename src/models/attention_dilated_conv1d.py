import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import MultiScaleCNNBranch, SkipConnectionMLP, AdaptiveFeaturePooling

@register_model("attention_dilated_conv1d")
class AttentionDilatedConv1D(BaseModel):
    """Middle-tier model that incorporates signal processing knowledge through multi-scale 
    convolutions and attention, without requiring heavy physics-based priors."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Multi-scale convolution branch
        self.multi_scale_branch = MultiScaleCNNBranch(
            in_channels=1,
            channels_per_kernel=16,
            kernel_sizes=(7, 15, 31),
            pooling=None
        )
        
        # Calculate number of channels from multi_scale_branch
        time_branch_channels = 16 * 3  # channels_per_kernel * len(kernel_sizes)
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(time_branch_channels, time_branch_channels // 2),
            nn.ReLU(),
            nn.Linear(time_branch_channels // 2, time_branch_channels),
            nn.Sigmoid()
        )
        
        # Dilated convolution layer for capturing wider temporal dependencies
        self.dilated_conv = nn.Conv1d(time_branch_channels, 64, kernel_size=5, padding=4, dilation=2)
        self.pool = nn.MaxPool1d(4)  # Combined pooling layers
        
        # Adaptive pooling to fix output size
        self.adaptive_pool = AdaptiveFeaturePooling(output_size=16, pooling_type='avg')
        
        # Fully connected layers with skip connection
        self.fc_block = SkipConnectionMLP(
            input_dim=64 * 16,
            hidden_dim=256,
            output_dim=64,
            dropout_rate=0.3
        )
        
        # Output layer
        self.output = nn.Linear(64, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Multi-scale time domain processing
        x = self.multi_scale_branch(x)
        
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights.view(batch_size, -1, 1)
        
        # Dilated convolution and pooling
        x = F.relu(self.dilated_conv(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        
        # Flatten and process through FC layers
        x = x.view(batch_size, -1)
        x = self.fc_block(x)
        
        # Output with sigmoid activation
        x = self.output(x)
        return torch.sigmoid(x)