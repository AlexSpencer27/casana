import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import BoundedPeakOutput

@register_model("simplest_possible")
class SimplestPossible(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        # Simple two-layer architecture
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(config.signal.length, 64)  # First reduce dimensionality
        self.fc2 = nn.Linear(64, 3)      # Output the required 3 values
        self.dropout = nn.Dropout(0.2)    # Light regularization
        
        # Use the new layer for bounded output
        self.peak_output = BoundedPeakOutput()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = self.flatten(x)
        
        # Simple forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply bounded output layer (handles sigmoid + midpoint)
        x = self.peak_output(x)
        
        return x