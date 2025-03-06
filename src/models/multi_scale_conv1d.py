import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel

@register_model("multi_scale_conv1d")
class MultiScaleConv1D(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Multi-scale convolution layers
        self.conv1_small = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv1_medium = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        self.conv1_large = nn.Conv1d(1, 16, kernel_size=31, padding=15)
        self.pool1 = nn.MaxPool1d(2)
        
        # Second convolution layer after concatenation
        self.conv2 = nn.Conv1d(48, 64, kernel_size=9, padding=4)
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
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Multi-scale time domain processing
        t_small = self.conv1_small(x)
        t_medium = self.conv1_medium(x)
        t_large = self.conv1_large(x)
        
        # Concatenate multi-scale features
        x = torch.cat([t_small, t_medium, t_large], dim=1)
        x = self.pool1(x)
        
        # Second convolution block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Third convolution block
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Flatten features
        x = x.view(batch_size, -1)
        
        # Main path through FC layers
        main = F.relu(self.bn1(self.fc1(x)))
        main = self.dropout(main)
        main = F.relu(self.bn2(self.fc2(main)))
        
        # Skip connection
        skip = F.relu(self.fc_skip(x))
        
        # Combine main path and skip connection
        x = main + skip
        x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        
        # Ensure peak1 < midpoint < peak2 using soft constraints
        sorted_x = x + 0.1 * (torch.sort(x, dim=1)[0] - x)
        
        return sorted_x 