import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses import register_loss
from src.losses.base_loss import BaseLoss

@register_loss("simple_mse")
class SimpleMSELoss(BaseLoss):
    """Mean Squared Error loss function that compares both positions and signal magnitudes"""
    
    def __init__(self, position_weight: float = 1.0, magnitude_weight: float = 0.5) -> None:
        """Initialize the loss function.
        
        Args:
            position_weight: Weight for the position loss component
            magnitude_weight: Weight for the magnitude loss component
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.position_weight = position_weight
        self.magnitude_weight = magnitude_weight
    
    def sample_signal_values(self, signals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Sample signal values at given positions using linear interpolation.
        
        Args:
            signals: Input signals of shape (batch_size, 1, signal_length)
            positions: Positions to sample at, normalized to [0,1] (batch_size, 3)
            
        Returns:
            Sampled values of shape (batch_size, 3)
        """
        batch_size, _, signal_length = signals.shape
        
        # Convert normalized positions to actual indices
        indices = positions * (signal_length - 1)
        
        # Get integer indices for interpolation
        idx_left = torch.floor(indices).long()
        idx_right = torch.ceil(indices).long()
        
        # Ensure indices are within bounds
        idx_left = torch.clamp(idx_left, 0, signal_length - 1)
        idx_right = torch.clamp(idx_right, 0, signal_length - 1)
        
        # Get weights for interpolation
        weights_right = indices - idx_left.float()
        weights_left = 1.0 - weights_right
        
        # Sample values and interpolate
        signals_flat = signals.squeeze(1)  # Remove channel dimension
        values_left = torch.gather(signals_flat, 1, idx_left)
        values_right = torch.gather(signals_flat, 1, idx_right)
        
        # Compute interpolated values
        interpolated = values_left * weights_left + values_right * weights_right
        
        return interpolated
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss between outputs and targets.
        
        Args:
            outputs: Model outputs (batch_size, 3) - [peak1, midpoint, peak2]
            targets: Ground truth targets (batch_size, 3) - [peak1, midpoint, peak2]
            signals: Input signals (batch_size, 1, signal_length)
            
        Returns:
            Combined loss value
        """
        # Position loss
        position_loss = self.mse(outputs, targets)
        
        # Magnitude loss
        target_magnitudes = self.sample_signal_values(signals, targets)
        predicted_magnitudes = self.sample_signal_values(signals, outputs)
        magnitude_loss = self.mse(predicted_magnitudes, target_magnitudes)
        
        # Combine losses
        total_loss = self.position_weight * position_loss + self.magnitude_weight * magnitude_loss
        
        return total_loss 