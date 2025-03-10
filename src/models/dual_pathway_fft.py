import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import (
    MultiScaleCNNBranch, 
    SpectralBranch, 
    CrossPathwayAttention, 
    SkipConnectionMLP, 
    PeakOrderingLayer,
    AdaptiveFeaturePooling
)

@register_model("dual_pathway_fft")
class DualPathwayFFT(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Time domain path with multi-scale kernels
        self.time_branch = MultiScaleCNNBranch(
            in_channels=1,
            channels_per_kernel=16,
            kernel_sizes=(7, 15, 31),
            pooling='max'
        )
        
        self.conv2 = nn.Conv1d(48, 64, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        
        # Adaptive pooling to fix output size
        self.adaptive_pool = AdaptiveFeaturePooling(output_size=16, pooling_type='avg')
        
        # Frequency domain path using SpectralBranch
        self.freq_branch = SpectralBranch(
            signal_length=config.signal.length,
            out_features=64,
            window_size=256,
            stride=128,
            process_complex='separate'
        )
        
        # Calculate feature dimensions
        time_features_dim = 64 * 16  # channels * adaptive_pool_size
        freq_features_dim = 64       # from spectral branch output
        
        # Cross-pathway attention for feature fusion
        self.attention = CrossPathwayAttention(
            time_features=time_features_dim,
            freq_features=freq_features_dim
        )
        
        # Fully connected layers with skip connection
        self.fc_block = SkipConnectionMLP(
            input_dim=time_features_dim + freq_features_dim,
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
        
        # Time domain processing
        time_features = self.time_branch(x)
        time_features = F.relu(self.conv2(time_features))
        time_features = self.pool2(time_features)
        time_features = F.relu(self.conv3(time_features))
        time_features = self.adaptive_pool(time_features)
        
        # Flatten time features
        time_features_flat = time_features.view(batch_size, -1)
        
        # Frequency domain processing
        freq_features = self.freq_branch(x)
        
        # Apply cross-pathway attention
        weighted_features = self.attention(time_features_flat, freq_features)
        
        # Process through FC layers with skip connection
        x = self.fc_block(weighted_features)
        
        # Output layer
        x = self.output(x)
        
        # Ensure peak1 < midpoint < peak2
        x = self.peak_ordering(x)
        
        return x