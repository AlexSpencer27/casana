import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """Base class for all loss functions"""
    
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
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
        signals_flat = signals.squeeze(1)
        values_left = torch.gather(signals_flat, 1, idx_left)
        values_right = torch.gather(signals_flat, 1, idx_right)
        
        # Compute interpolated values
        interpolated = values_left * weights_left + values_right * weights_right
        
        return interpolated
    
    def compute_position_and_magnitude_losses(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        signals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute position and magnitude losses.
        
        Args:
            outputs: Model outputs (batch_size, 3) - [peak1, midpoint, peak2]
            targets: Ground truth targets (batch_size, 3) - [peak1, midpoint, peak2]
            signals: Input signals (batch_size, 1, signal_length)
            
        Returns:
            Tuple of (position_loss, magnitude_loss)
        """
        # Position loss
        position_loss = self.mse(outputs, targets)
        
        # Magnitude loss
        target_magnitudes = self.sample_signal_values(signals, targets)
        predicted_magnitudes = self.sample_signal_values(signals, outputs)
        magnitude_loss = self.mse(predicted_magnitudes, target_magnitudes)
        
        return position_loss, magnitude_loss
    
    @abstractmethod
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function
        
        Args:
            outputs: Model outputs (batch_size, 3) - [peak1, midpoint, peak2]
            targets: Ground truth targets (batch_size, 3) - [peak1, midpoint, peak2]
            signals: Input signals (batch_size, 1, signal_length)
            
        Returns:
            Loss value
        """
        pass 