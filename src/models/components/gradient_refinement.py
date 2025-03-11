"""
Gradient refinement module for improving peak predictions using signal gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientRefinementModule(nn.Module):
    def __init__(self, signal_length, base_step_size=0.01, max_iterations=15):
        """
        Initialize gradient refinement module.
        
        Args:
            signal_length: Length of input signal
            base_step_size: Base learning rate for gradient updates (default: 0.01)
            max_iterations: Maximum number of refinement iterations (default: 15)
        """
        super().__init__()
        self.signal_length = signal_length
        self.base_step_size = base_step_size
        self.max_iterations = max_iterations
        
        # Create position indices for gradient calculation
        pos_indices = torch.arange(signal_length, dtype=torch.float32)
        self.register_buffer('pos_indices', pos_indices)
    
    def _sample_signal_values(self, signals, positions):
        """
        Sample signal values at given positions using linear interpolation.
        
        Args:
            signals: Input signals of shape [batch_size, signal_length]
            positions: Positions to sample at of shape [batch_size, num_peaks]
            
        Returns:
            Sampled values of shape [batch_size, num_peaks]
        """
        batch_size = signals.shape[0]
        device = signals.device
        
        # Scale positions to signal length and clamp to valid range
        positions = torch.clamp(positions * (self.signal_length - 1), 0, self.signal_length - 1)
        
        # Get indices for left and right samples
        idx_left = positions.floor().long()
        idx_right = positions.ceil().long()
        
        # Double-check indices are within bounds (redundant but safe)
        idx_left = torch.clamp(idx_left, 0, self.signal_length - 1)
        idx_right = torch.clamp(idx_right, 0, self.signal_length - 1)
        
        # Create batch indices for gathering
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, positions.size(1))
        
        # Reshape indices for gather operation
        batch_indices = batch_indices.contiguous()
        idx_left = idx_left.contiguous()
        idx_right = idx_right.contiguous()
        
        # Sample values using gathered indices
        values_left = signals[batch_indices.view(-1), idx_left.view(-1)].view(batch_size, -1)
        values_right = signals[batch_indices.view(-1), idx_right.view(-1)].view(batch_size, -1)
        
        # Linear interpolation weights
        weights_right = positions - idx_left.float()
        weights_left = 1.0 - weights_right
        
        # Interpolate values
        interpolated = weights_left * values_left + weights_right * values_right
        return interpolated
    
    def _calculate_gradients_and_curvatures(self, signals, positions):
        """
        Calculate gradients and curvatures at peak positions with multi-scale approach.
        
        Args:
            signals: Input signals of shape [batch_size, signal_length]
            positions: Peak positions of shape [batch_size, num_peaks]
            
        Returns:
            Tuple of (gradients, curvatures) each of shape [batch_size, num_peaks]
        """
        # Use multiple scales for gradient calculation
        eps_scales = [0.001, 0.003, 0.01]  # Multiple scales for robustness
        gradients_multi = []
        curvatures_multi = []
        
        for eps in eps_scales:
            # Ensure positions are within valid range for offset calculations
            positions_clamped = torch.clamp(positions, eps, 1.0 - eps)
            
            # Sample values at current positions and offsets
            values = self._sample_signal_values(signals, positions_clamped)
            values_left = self._sample_signal_values(signals, positions_clamped - eps)
            values_right = self._sample_signal_values(signals, positions_clamped + eps)
            
            # Calculate gradients using central difference
            gradients = (values_right - values_left) / (2 * eps)
            
            # Calculate curvatures using second derivative
            curvatures = (values_right + values_left - 2 * values) / (eps * eps)
            
            gradients_multi.append(gradients)
            curvatures_multi.append(curvatures)
        
        # Combine multi-scale gradients and curvatures (weighted average favoring smaller scales)
        weights = torch.tensor([0.5, 0.3, 0.2], device=positions.device)
        gradients = sum(g * w for g, w in zip(gradients_multi, weights))
        curvatures = sum(c * w for c, w in zip(curvatures_multi, weights))
        
        # Add small epsilon to curvatures for numerical stability
        curvatures = torch.clamp(curvatures, min=-1e3, max=1e3)
        
        return gradients, curvatures
    
    def forward(self, initial_predictions, signals):
        """
        Refine peak predictions using gradient information with adaptive step sizes.
        
        Args:
            initial_predictions: Initial peak predictions of shape [batch_size, 3]
            signals: Input signals of shape [batch_size, signal_length] or [batch_size, 1, signal_length]
            
        Returns:
            Refined predictions of shape [batch_size, 3]
        """
        # Ensure signals has correct shape [batch_size, signal_length]
        if signals.dim() == 3:
            signals = signals.squeeze(1)
            
        # Initialize refined positions with initial predictions
        positions = initial_predictions.clone()
        
        # Track previous positions and best positions
        prev_positions = positions.clone()
        best_positions = positions.clone()
        best_values = self._sample_signal_values(signals, positions)
        
        # Adaptive step size parameters
        step_size = self.base_step_size
        step_decay = 0.9
        min_step = 1e-4
        
        # Iterative refinement
        for iter_idx in range(self.max_iterations):
            # Calculate gradients and curvatures
            gradients, curvatures = self._calculate_gradients_and_curvatures(signals, positions)
            
            # Update positions using Newton's method with adaptive step size
            step = -gradients / (torch.abs(curvatures) + 1e-6)
            
            # Use larger step bounds for early iterations, smaller for later
            max_step = 0.2 * (step_decay ** iter_idx)
            step = torch.clamp(step, -max_step, max_step)
            
            # Apply step size and update positions
            positions = positions + step_size * step
            
            # Clamp positions to [0, 1] range
            positions = torch.clamp(positions, 0.0, 1.0)
            
            # Update best positions if current values are higher
            current_values = self._sample_signal_values(signals, positions)
            
            # Handle peak1, midpoint, and peak2 separately
            # For peak1 and peak2, we want higher values
            # For midpoint, we'll keep it as the average of its neighbors
            improved_mask_peaks = current_values > best_values
            
            # Update peak positions where improved
            best_positions = positions.clone()  # Start with current positions
            
            # Only update where values improved
            mask = improved_mask_peaks
            best_positions = torch.where(mask, positions, best_positions)
            
            # Update best values
            best_values = torch.where(improved_mask_peaks, current_values, best_values)
            
            # Check for convergence with relaxed criterion
            pos_change = torch.abs(positions - prev_positions).max()
            if pos_change < 1e-4 or step_size < min_step:
                break
                
            # Update step size and previous positions
            step_size *= step_decay
            prev_positions = positions.clone()
        
        # Return best found positions
        return best_positions.to(initial_predictions.device) 