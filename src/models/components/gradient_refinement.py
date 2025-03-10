"""
Gradient refinement module for peak position refinement using gradient information.
"""

import torch
import torch.nn as nn
import math

class GradientRefinementModule(nn.Module):
    def __init__(self, signal_length, base_step_size=0.002, max_iterations=50):
        """
        Refine peak positions using gradient information.
        
        Args:
            signal_length: Signal length
            base_step_size: Step size for gradient updates (default: 0.002)
            max_iterations: Maximum number of refinement iterations (default: 50)
        """
        super().__init__()
        self.signal_length = signal_length
        self.base_step_size = base_step_size
        self.max_iterations = max_iterations
        self.min_delta = 1e-4  # Convergence criterion
        self.derivative_step = 0.01  # 1% of normalized range
        
        # Fixed parameters with good defaults
        self.momentum = 0.8  # Lower momentum for noisy signals
        self.trust_region_radius = 0.15  # Allows movement while preventing jumps
        
        # Window size based on signal characteristics (peaks are 2-8% of signal)
        window_size_proportion = 0.08  # Match maximum peak width
        min_window_size = 10  # Minimum peak width
        self.window_size = max(min_window_size, int(signal_length * window_size_proportion))

    def _sample_signal_values(self, signals, positions):
        """
        Sample signal values at given positions using linear interpolation.
        
        Args:
            signals: Input signals of shape (batch_size, signal_length)
            positions: Positions to sample at, normalized to [0,1] (batch_size,)
            
        Returns:
            Sampled values of shape (batch_size,)
        """
        indices = positions * (signals.shape[1] - 1)
        idx_left = torch.floor(indices).long()
        idx_right = torch.ceil(indices).long()
        idx_left = torch.clamp(idx_left, 0, signals.shape[1] - 1)
        idx_right = torch.clamp(idx_right, 0, signals.shape[1] - 1)
        weights_right = indices - idx_left.float()
        weights_left = 1.0 - weights_right
        values_left = signals[torch.arange(signals.shape[0]), idx_left]
        values_right = signals[torch.arange(signals.shape[0]), idx_right]
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
        pos_left = torch.clamp(positions - self.derivative_step, 0, 1)
        pos_right = torch.clamp(positions + self.derivative_step, 0, 1)
        values = self._sample_signal_values(signal, positions)
        values_left = self._sample_signal_values(signal, pos_left)
        values_right = self._sample_signal_values(signal, pos_right)
        gradients = (values_right - values_left) / (2 * self.derivative_step)
        curvatures = (values_right - 2 * values + values_left) / (self.derivative_step * self.derivative_step)
        return gradients, curvatures
    
    def forward(self, signal, peak_positions):
        """
        Refine peak positions using adaptive gradient information.
        
        Args:
            signal: Input signal of shape [batch_size, signal_length]
            peak_positions: Initial peak positions of shape [batch_size, 3]
                           with values in [0, 1] representing normalized positions
                           
        Returns:
            Refined peak positions of shape [batch_size, 3]
        """
        batch_size = signal.shape[0]
        current_positions = peak_positions.clone()
        
        # Track convergence for each peak separately
        converged = torch.zeros((batch_size, 2), dtype=torch.bool, device=signal.device)
        iteration = 0
        
        while not converged.all() and iteration < self.max_iterations:
            # Calculate gradients and curvatures for non-converged peaks
            updates = torch.zeros_like(current_positions)
            
            for peak_idx, pos_idx in [(0, 0), (1, 2)]:  # Peak indices 0 and 2 (1 is midpoint)
                # Only update non-converged peaks
                active_peaks = ~converged[:, peak_idx]
                if active_peaks.any():
                    gradients, curvatures = self._calculate_gradients_and_curvatures(
                        signal[active_peaks], current_positions[active_peaks, pos_idx]
                    )
                    
                    # Update positions using gradient information
                    peak_updates = -self.base_step_size * gradients * (curvatures < 0).float()
                    updates[active_peaks, pos_idx] = peak_updates
                    
                    # Check convergence for this peak
                    converged[active_peaks, peak_idx] = torch.abs(peak_updates) < self.min_delta
            
            # Update midpoint
            updates[:, 1] = (updates[:, 0] + updates[:, 2]) / 2
            
            # Apply updates with bounds checking
            new_positions = torch.clamp(current_positions + updates, 0.0, 1.0)
            current_positions = new_positions
            iteration += 1
        
        # Ensure proper ordering
        is_ordered = current_positions[:, 0] < current_positions[:, 2]
        swapped = torch.stack([
            current_positions[:, 2],
            (current_positions[:, 0] + current_positions[:, 2]) / 2,
            current_positions[:, 0]
        ], dim=1)
        
        return torch.where(is_ordered.unsqueeze(1), current_positions, swapped) 