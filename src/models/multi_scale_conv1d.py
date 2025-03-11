import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components.multi_scale_cnn import MultiScaleCNNBranch

@register_model("multi_scale_conv1d")
class MultiScaleConv1D(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize the multi-scale CNN branch with default parameters
        self.multi_scale_cnn = MultiScaleCNNBranch(
            in_channels=1,
            channels_per_kernel=16,
            kernel_sizes=(7, 15, 31),
            pooling='max',
            pooling_size=2,
            activation=F.relu,
            dropout_rate=0.3
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the multi-scale CNN branch
        return self.multi_scale_cnn(x)
        
        
        