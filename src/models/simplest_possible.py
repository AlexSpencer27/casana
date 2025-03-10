import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import PeakOrderingLayer

@register_model("simplest_possible")
class SimplestPossible(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        # Simple two-layer architecture
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(config.signal.length, 64)  # First reduce dimensionality
        self.fc2 = nn.Linear(64, 3)      # Output the required 3 values
        self.dropout = nn.Dropout(0.2)    # Light regularization
        
        # Peak ordering layer
        self.peak_ordering = PeakOrderingLayer()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = self.flatten(x)
        
        # Simple forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Ensure peak1 < midpoint < peak2 using component
        x = self.peak_ordering(x)
        
        return x