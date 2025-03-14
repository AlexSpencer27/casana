import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel

@register_model("simplest_possible")
class SimplestPossible(BaseModel):
    """The most basic model that makes minimal assumptions about the signal structure.
    Uses a simple two-layer fully connected architecture with dropout regularization."""
    
    def __init__(self) -> None:
        super().__init__()
        # Simple two-layer architecture
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(config.signal.length, 64)  # First reduce dimensionality
        self.fc2 = nn.Linear(64, 3)      # Output the required 3 values
        self.dropout = nn.Dropout(0.2)    # Light regularization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = self.flatten(x)
        
        # Simple forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x