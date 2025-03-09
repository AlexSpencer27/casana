"""
Gradient refinement module for peak position refinement using gradient information.
"""

import torch
import torch.nn as nn

class GradientRefinementModule(nn.Module):
    def __init__(self, signal_length=2048, num_iterations=3, step_size=0.001):
        """
        Refine peak positions using gradient information.
        
        Args:
            signal_length: Signal length (default: 2048)
            num_iterations: Number of refinement iterations (default: 3)
            step_size: Adjustment step size (default: 0.001)
        """
        super().__init__()
        self.signal_length = signal_length
        self.num_iterations = num_iterations
        self.step_size = step_size
        # Step size based on quarter of minimum peak width (10/4 = 2.5 samples)
        self.derivative_step = (10.0 / 4.0) / signal_length
    
    def _sample_signal_values(self, signals, positions):
        """
        Sample signal values at given positions using linear interpolation.
        
        Args:
            signals: Input signals of shape (batch_size, signal_length)
            positions: Positions to sample at, normalized to [0,1] (batch_size,)
            
        Returns:
            Sampled values of shape (batch_size,)
        """
        # Convert normalized positions to indices
        indices = positions * (signals.shape[1] - 1)
        
        # Get integer indices for interpolation
        idx_left = torch.floor(indices).long()
        idx_right = torch.ceil(indices).long()
        
        # Ensure indices are within bounds
        idx_left = torch.clamp(idx_left, 0, signals.shape[1] - 1)
        idx_right = torch.clamp(idx_right, 0, signals.shape[1] - 1)
        
        # Get weights for interpolation
        weights_right = indices - idx_left.float()
        weights_left = 1.0 - weights_right
        
        # Sample values and interpolate
        values_left = signals[torch.arange(signals.shape[0]), idx_left]
        values_right = signals[torch.arange(signals.shape[0]), idx_right]
        
        # Compute interpolated values
        return values_left * weights_left + values_right * weights_right
    
    def _calculate_gradients_and_curvatures(self, signal, positions):
        """
        Calculate gradients and curvatures at specified positions using interpolation.
        
        Args:
            signal: Tensor of shape [batch_size, signal_length]
            positions: Tensor of shape [batch_size] with values in [0, 1]
            
        Returns:
            gradients: Tensor of shape [batch_size]
            curvatures: Tensor of shape [batch_size]
        """
        # Sample points for central difference
        pos_left = torch.clamp(positions - self.derivative_step, 0, 1)
        pos_right = torch.clamp(positions + self.derivative_step, 0, 1)
        
        # Get values at all points using interpolation
        values = self._sample_signal_values(signal, positions)
        values_left = self._sample_signal_values(signal, pos_left)
        values_right = self._sample_signal_values(signal, pos_right)
        
        # Compute first derivative using central difference
        gradients = (values_right - values_left) / (2 * self.derivative_step)
        
        # Compute second derivative using central difference
        curvatures = (values_right - 2 * values + values_left) / (self.derivative_step * self.derivative_step)
        
        return gradients, curvatures
    
    def forward(self, signal, peak_positions):
        """
        Refine peak positions using gradient information.
        
        Args:
            signal: Input signal of shape [batch_size, signal_length]
            peak_positions: Initial peak positions of shape [batch_size, 3]
                           with values in [0, 1] representing normalized positions
                           
        Returns:
            Refined peak positions of shape [batch_size, 3]
        """
        batch_size = signal.shape[0]
        current_positions = peak_positions
        
        for _ in range(self.num_iterations):
            # Calculate gradients and curvatures
            p1_gradients, p1_curvatures = self._calculate_gradients_and_curvatures(signal, current_positions[:, 0])
            p2_gradients, p2_curvatures = self._calculate_gradients_and_curvatures(signal, current_positions[:, 2])
            
            # Calculate adjustments
            p1_adjustment = -self.step_size * p1_gradients * (p1_curvatures < 0).float()
            p2_adjustment = -self.step_size * p2_gradients * (p2_curvatures < 0).float()
            
            # Create new position tensors using torch.stack
            p1_new = torch.clamp(current_positions[:, 0] + p1_adjustment, 0.0, 1.0)
            p2_new = torch.clamp(current_positions[:, 2] + p2_adjustment, 0.0, 1.0)
            mid_new = (p1_new + p2_new) / 2
            
            current_positions = torch.stack([p1_new, mid_new, p2_new], dim=1)
        
        # Handle ordering without in-place operations
        is_ordered = current_positions[:, 0] < current_positions[:, 2]
        
        # Create swapped version
        p1_swapped = current_positions[:, 2]
        p2_swapped = current_positions[:, 0]
        mid_swapped = (p1_swapped + p2_swapped) / 2
        swapped = torch.stack([p1_swapped, mid_swapped, p2_swapped], dim=1)
        
        # Select between original and swapped using where
        return torch.where(is_ordered.unsqueeze(1), current_positions, swapped) 