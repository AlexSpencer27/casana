import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses import register_loss
from src.losses.base_loss import BaseLoss
from src.config.config import config

@register_loss("peak_loss")
class PeakLoss(BaseLoss):
    """Unified loss function for peak detection that can handle both simple MSE and gradient-aware cases.
    When gradient_weight and second_derivative_weight are 0, behaves like SimpleMSELoss.
    When non-zero, adds gradient and curvature constraints for more physics-informed behavior."""
    
    def __init__(self) -> None:
        """Initialize the loss function."""
        super().__init__()
        # Get initial weights from curriculum
        self.position_weight = config.curriculum.loss_weights.start.position
        self.magnitude_weight = config.curriculum.loss_weights.start.magnitude
        self.gradient_weight = config.curriculum.loss_weights.start.gradient
        self.second_derivative_weight = config.curriculum.loss_weights.start.second_derivative
        
        # Only compute step size if we're using gradient terms
        if self.gradient_weight > 0 or self.second_derivative_weight > 0:
            # Step size based on quarter of standard peak width (25/4 = 6.25 samples)
            self.step_size = (25.0 / 4.0) / config.signal.length
    
    def sample_signal_values(self, signals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Sample signal values at given positions using linear interpolation."""
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
        
        return values_left * weights_left + values_right * weights_right
    
    def compute_gradients(self, signals: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute first and second derivatives at given positions."""
        # Sample points for central difference
        pos_left = torch.clamp(positions - self.step_size, 0, 1)
        pos_right = torch.clamp(positions + self.step_size, 0, 1)
        
        # Get values at all points
        values = self.sample_signal_values(signals, positions)
        values_left = self.sample_signal_values(signals, pos_left)
        values_right = self.sample_signal_values(signals, pos_right)
        
        # Compute derivatives using central difference
        first_derivative = (values_right - values_left) / (2 * self.step_size)
        second_derivative = (values_right - 2 * values + values_left) / (self.step_size * self.step_size)
        
        return first_derivative, second_derivative
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss between outputs and targets.
        
        Args:
            outputs: Model outputs (batch_size, 3) - [peak1, midpoint, peak2]
            targets: Ground truth targets (batch_size, 3) - [peak1, midpoint, peak2]
            signals: Input signals (batch_size, 1, signal_length)
            
        Returns:
            Combined loss value based on enabled terms
        """
        # Get position and magnitude losses (always computed)
        position_loss, magnitude_loss = self.compute_position_and_magnitude_losses(outputs, targets, signals)
        total_loss = self.position_weight * position_loss + self.magnitude_weight * magnitude_loss
        
        # Only compute gradient terms if weights are non-zero
        if self.gradient_weight > 0 or self.second_derivative_weight > 0:
            # Compute gradients at predicted peak positions (not midpoint)
            peak_positions = torch.cat([outputs[:, 0:1], outputs[:, 2:]], dim=1)
            first_deriv, second_deriv = self.compute_gradients(signals, peak_positions)
            
            if self.gradient_weight > 0:
                # Gradient loss - first derivative should be zero at peaks
                gradient_loss = torch.mean(first_deriv ** 2)
                total_loss += self.gradient_weight * gradient_loss
            
            if self.second_derivative_weight > 0:
                # Second derivative loss - maximize negative curvature at peaks
                curvature_error = -torch.mean(second_deriv)
                total_loss += self.second_derivative_weight * curvature_error

        # normalize loss by sum of weights
        total_loss /= (self.position_weight + self.magnitude_weight + self.gradient_weight + self.second_derivative_weight)
        
        return total_loss 