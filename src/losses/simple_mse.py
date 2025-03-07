import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses import register_loss
from src.losses.base_loss import BaseLoss
from src.config.config import config

@register_loss("simple_mse")
class SimpleMSELoss(BaseLoss):
    """Mean Squared Error loss function that compares both positions and signal magnitudes"""
    
    def __init__(self) -> None:
        """Initialize the loss function."""
        super().__init__()
        self.position_weight = config.loss.position_weight
        self.magnitude_weight = config.loss.magnitude_weight
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss between outputs and targets.
        
        Args:
            outputs: Model outputs (batch_size, 3) - [peak1, midpoint, peak2]
            targets: Ground truth targets (batch_size, 3) - [peak1, midpoint, peak2]
            signals: Input signals (batch_size, 1, signal_length)
            
        Returns:
            Combined loss value
        """
        # Get position and magnitude losses from base class
        position_loss, magnitude_loss = self.compute_position_and_magnitude_losses(outputs, targets, signals)
        
        # Combine losses
        total_loss = self.position_weight * position_loss + self.magnitude_weight * magnitude_loss
        
        return total_loss 