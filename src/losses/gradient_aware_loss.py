import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses import register_loss
from src.losses.base_loss import BaseLoss
from src.config.config import config

@register_loss("gradient_aware")
class GradientAwareLoss(BaseLoss):
    """Loss function that checks both signal values and gradient properties at peak locations"""
    
    def __init__(self) -> None:
        """Initialize the loss function."""
        super().__init__()
        self.position_weight = config.loss.position_weight
        self.magnitude_weight = config.loss.magnitude_weight
        self.gradient_weight = config.loss.gradient_weight
        self.second_derivative_weight = config.loss.second_derivative_weight
        
        # Step size based on quarter of minimum peak width (10/4 = 2.5 samples)
        # This gives us a good balance between local behavior and numerical stability
        # Convert to normalized coordinates by dividing by signal length
        self.step_size = (10.0 / 4.0) / config.signal.length
    
    def sample_signal_values(self, signals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Sample signal values at given positions using linear interpolation.
        
        Args:
            signals: Input signals of shape (batch_size, 1, signal_length)
            positions: Positions to sample at, normalized to [0,1] (batch_size, 3)
            
        Returns:
            Sampled values of shape (batch_size, 3)
        """
        # Convert normalized positions to indices
        indices = positions * (signals.size(-1) - 1)
        
        # Get integer indices for interpolation
        idx_left = torch.floor(indices).long()
        idx_right = torch.ceil(indices).long()
        
        # Ensure indices are within bounds
        idx_left = torch.clamp(idx_left, 0, signals.size(-1) - 1)
        idx_right = torch.clamp(idx_right, 0, signals.size(-1) - 1)
        
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
    
    def compute_gradients(self, signals: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute first and second derivatives at given positions.
        
        Args:
            signals: Input signals of shape (batch_size, 1, signal_length)
            positions: Positions to compute gradients at, normalized to [0,1] (batch_size, 3)
            
        Returns:
            Tuple of (first_derivatives, second_derivatives) at the specified positions
        """
        # Sample points for central difference
        pos_left = torch.clamp(positions - self.step_size, 0, 1)
        pos_right = torch.clamp(positions + self.step_size, 0, 1)
        
        # Get values at all points
        values = self.sample_signal_values(signals, positions)
        values_left = self.sample_signal_values(signals, pos_left)
        values_right = self.sample_signal_values(signals, pos_right)
        
        # Note: signals are already normalized in generate_batch()
        # signal = (signal - signal.mean()) / signal.std()
        # So we don't need additional normalization here
        
        # Compute first derivative using central difference
        first_derivative = (values_right - values_left) / (2 * self.step_size)
        
        # Compute second derivative using central difference
        second_derivative = (values_right - 2 * values + values_left) / (self.step_size * self.step_size)
        
        return first_derivative, second_derivative
    
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
        
        # Compute gradients at predicted peak positions
        # We only care about peak positions (indices 0 and 2), not midpoint
        peak_positions = torch.cat([outputs[:, 0:1], outputs[:, 2:]], dim=1)
        first_deriv, second_deriv = self.compute_gradients(signals, peak_positions)
        
        # Gradient loss - first derivative should be zero at peaks
        gradient_loss = torch.mean(first_deriv ** 2)
        
        # Second derivative loss - should be negative at peaks
        second_deriv_loss = torch.mean(F.relu(second_deriv))
        
        # Note: No need to normalize by signal variance since signals are already normalized
        
        # Combine all losses
        total_loss = (
            self.position_weight * position_loss +
            self.magnitude_weight * magnitude_loss +
            self.gradient_weight * gradient_loss +
            self.second_derivative_weight * second_deriv_loss
        )
        
        return total_loss 