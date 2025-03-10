"""
Gradient refinement module for peak position refinement using gradient information.
"""

import torch
import torch.nn as nn
import math

class GradientRefinementModule(nn.Module):
    def __init__(self, signal_length, base_step_size=0.002, max_iterations=7):
        """
        Refine peak positions using adaptive gradient information.
        
        Args:
            signal_length: Signal length
            base_step_size: Base learning rate for updates (default: 0.002)
            max_iterations: Maximum number of refinement iterations (default: 7)
        """
        super().__init__()
        self.signal_length = signal_length
        self.base_step_size = base_step_size
        self.max_iterations = max_iterations
        
        # Fixed parameters with good defaults
        self.momentum = 0.8  # Lower momentum for noisy signals
        self.min_delta = 1e-4  # Convergence criterion matching noise floor
        self.trust_region_radius = 0.15  # Allows movement while preventing jumps
        
        # Window size based on signal characteristics (peaks are 2-8% of signal)
        window_size_proportion = 0.08  # Match maximum peak width
        min_window_size = 10  # Minimum peak width
        self.window_size = max(min_window_size, int(signal_length * window_size_proportion))
        
        # Derivative step based on minimum peak width
        self.derivative_step = (min_window_size / signal_length) / 4.0

    def _estimate_local_snr(self, signal, positions):
        """
        Estimate local signal-to-noise ratio around peak positions.
        
        Args:
            signal: Input signals of shape (batch_size, signal_length)
            positions: Positions to analyze, normalized to [0,1] (batch_size,)
            
        Returns:
            SNR estimates of shape (batch_size,)
        """
        batch_size = signal.shape[0]
        window_radius = min(self.window_size // 2, signal.shape[1] // 4)  # Ensure window isn't too large
        
        # Convert positions to indices
        indices = (positions * (signal.shape[1] - 1)).long()
        
        # Calculate local windows
        snr_estimates = torch.zeros(batch_size, device=signal.device)
        for i in range(batch_size):
            # Get window boundaries with safety checks
            center_idx = indices[i]
            start_idx = max(0, center_idx - window_radius)
            end_idx = min(signal.shape[1], center_idx + window_radius)
            
            # Ensure window has at least 2 samples
            if end_idx - start_idx < 2:
                # If window is too small, expand it symmetrically
                half_width = max(1, (signal.shape[1] - 1) // 8)  # Use 1/8 of signal length as minimum
                start_idx = max(0, center_idx - half_width)
                end_idx = min(signal.shape[1], center_idx + half_width)
            
            # Extract window
            window = signal[i, start_idx:end_idx]
            
            # Estimate peak value and noise level
            peak_value = window.max(dim=0)[0]  # Explicitly specify reduction dimension
            noise_std = torch.std(window, dim=0)  # Explicitly specify reduction dimension
            
            # Calculate SNR (avoid division by zero)
            snr_estimates[i] = peak_value / (noise_std + 1e-6)
        
        return torch.clamp(snr_estimates, min=0.1, max=10.0)

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
    
    def _calculate_adaptive_step_size(self, gradients, curvatures, snr):
        """
        Calculate adaptive step size based on local signal properties.
        
        Args:
            gradients: Gradient values of shape [batch_size]
            curvatures: Curvature values of shape [batch_size]
            snr: Signal-to-noise ratio estimates of shape [batch_size]
            
        Returns:
            Adaptive step sizes of shape [batch_size]
        """
        # Scale step size based on curvature and SNR
        curvature_factor = torch.clamp(torch.abs(curvatures), min=0.1, max=10.0)
        snr_factor = torch.exp(-1.0 / snr)
        
        step_size = self.base_step_size * snr_factor / curvature_factor
        
        # Clamp to trust region
        max_step = self.trust_region_radius / (torch.abs(gradients) + 1e-6)
        return torch.clamp(step_size, max=max_step)
    
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
        current_positions = peak_positions
        velocity = torch.zeros_like(peak_positions)
        
        # Initialize best positions and values
        best_positions = current_positions.clone()
        best_values = self._sample_signal_values(signal, best_positions[:, 0]) + \
                     self._sample_signal_values(signal, best_positions[:, 2])
        
        converged = torch.zeros(batch_size, dtype=torch.bool, device=signal.device)
        iteration = 0
        
        while not converged.all() and iteration < self.max_iterations:
            # Calculate gradients and curvatures for both peaks
            p1_gradients, p1_curvatures = self._calculate_gradients_and_curvatures(signal, current_positions[:, 0])
            p2_gradients, p2_curvatures = self._calculate_gradients_and_curvatures(signal, current_positions[:, 2])
            
            # Estimate local SNR
            p1_snr = self._estimate_local_snr(signal, current_positions[:, 0])
            p2_snr = self._estimate_local_snr(signal, current_positions[:, 2])
            
            # Calculate adaptive step sizes
            p1_step = self._calculate_adaptive_step_size(p1_gradients, p1_curvatures, p1_snr)
            p2_step = self._calculate_adaptive_step_size(p2_gradients, p2_curvatures, p2_snr)
            
            # Update velocities with momentum
            velocity[:, 0] = self.momentum * velocity[:, 0] - p1_step * p1_gradients * (p1_curvatures < 0).float()
            velocity[:, 2] = self.momentum * velocity[:, 2] - p2_step * p2_gradients * (p2_curvatures < 0).float()
            
            # Apply updates
            new_positions = current_positions.clone()
            new_positions[:, [0, 2]] += velocity[:, [0, 2]]
            new_positions[:, 1] = (new_positions[:, 0] + new_positions[:, 2]) / 2
            
            # Clamp positions to valid range
            new_positions = torch.clamp(new_positions, 0.0, 1.0)
            
            # Check convergence
            position_change = torch.norm(new_positions - current_positions, dim=1)
            converged = position_change < self.min_delta
            
            # Update best positions if better
            current_values = self._sample_signal_values(signal, new_positions[:, 0]) + \
                           self._sample_signal_values(signal, new_positions[:, 2])
            improved = current_values > best_values
            best_positions[improved] = new_positions[improved]
            best_values[improved] = current_values[improved]
            
            current_positions = new_positions
            iteration += 1
        
        # Use best found positions
        current_positions = best_positions
        
        # Handle ordering without in-place operations
        is_ordered = current_positions[:, 0] < current_positions[:, 2]
        
        # Create swapped version
        p1_swapped = current_positions[:, 2]
        p2_swapped = current_positions[:, 0]
        mid_swapped = (p1_swapped + p2_swapped) / 2
        swapped = torch.stack([p1_swapped, mid_swapped, p2_swapped], dim=1)
        
        # Select between original and swapped using where
        return torch.where(is_ordered.unsqueeze(1), current_positions, swapped) 