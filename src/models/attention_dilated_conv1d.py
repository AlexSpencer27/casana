import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import MultiScaleCNNBranch, SkipConnectionMLP, AttentionModule, AdaptiveFeaturePooling

@register_model("attention_dilated_conv1d")
class AttentionDilatedConv1D(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Multi-scale convolution using component
        self.multi_scale_branch = MultiScaleCNNBranch(
            in_channels=1,
            channels_per_kernel=16,
            kernel_sizes=(7, 15, 31),
            pooling=None,  # No pooling here as we'll apply attention first
            feature_extractor_mode=True
        )
        
        # Calculate number of channels from multi_scale_branch
        time_branch_channels = 16 * 3  # channels_per_kernel * len(kernel_sizes)
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling along sequence dimension
            nn.Flatten(),  # Flatten to [batch_size, channels]
            nn.Linear(time_branch_channels, time_branch_channels // 2),
            nn.ReLU(),
            nn.Linear(time_branch_channels // 2, time_branch_channels),
            nn.Sigmoid()
        )
        
        self.pool1 = nn.MaxPool1d(2)
        
        # Dilated convolution layer after concatenation
        self.dilated_conv = nn.Conv1d(time_branch_channels, 64, kernel_size=5, padding=4, dilation=2)
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Multi-scale time domain processing using component
        x = self.multi_scale_branch(x)  # [batch_size, time_branch_channels, seq_len]
        
        # Apply channel attention mechanism
        # First get channel attention weights
        channel_weights = self.channel_attention(x)  # [batch_size, time_branch_channels]
        # Reshape to [batch_size, channels, 1] for broadcasting
        channel_weights = channel_weights.view(batch_size, -1, 1)
        # Apply weights to each channel
        x = x * channel_weights
        
        x = self.pool1(x)
        
        # Dilated convolution block
        x = F.relu(self.dilated_conv(x))
        x = self.pool2(x)
        
        # Third convolution block
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Flatten features
        x = x.view(batch_size, -1)
        
        # Process through FC layers with skip connection
        x = self.fc_block(x)
        
        # Output layer with sigmoid activation
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x