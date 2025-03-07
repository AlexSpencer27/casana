import torch
import torch.nn as nn
from src.losses import register_loss
from src.losses.base_loss import BaseLoss

@register_loss("simple_mse")
class SimpleMSELoss(BaseLoss):
    """Simple Mean Squared Error loss function"""
    
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate MSE loss between outputs and targets
        
        Args:
            outputs: Model outputs (batch_size, 3) - [peak1, midpoint, peak2]
            targets: Ground truth targets (batch_size, 3) - [peak1, midpoint, peak2]
            
        Returns:
            MSE loss value
        """
        return self.mse(outputs, targets) 