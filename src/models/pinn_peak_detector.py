import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel

class HanningTemplateLayer(nn.Module):
    """Custom layer for template matching with Hanning windows of different widths"""
    def __init__(self, min_width=10, max_width=40, num_templates=4):
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.num_templates = num_templates
        
        # Generate template widths
        self.widths = torch.linspace(min_width, max_width, num_templates, dtype=torch.int)
        
        # Create learnable weights for each template
        self.template_weights = nn.Parameter(torch.ones(num_templates))
        
    def forward(self, x):
        batch_size, signal_length = x.shape
        all_correlations = []
        
        for i, width in enumerate(self.widths):
            width = width.item()
            template = torch.from_numpy(np.hanning(width * 2)).float().to(x.device)
            batch_correlations = []
            
            for b in range(batch_size):
                signal = x[b]
                correlations = []
                
                for j in range(signal_length):
                    start_idx = max(0, j - width)
                    end_idx = min(signal_length, j + width)
                    templ_start = max(0, width - j)
                    templ_end = min(2 * width, width + (signal_length - j))
                    
                    signal_segment = signal[start_idx:end_idx]
                    template_segment = template[templ_start:templ_end]
                    
                    min_length = min(len(signal_segment), len(template_segment))
                    if min_length > 0:
                        corr = torch.sum(signal_segment[:min_length] * template_segment[:min_length])
                    else:
                        corr = torch.tensor(0.0).to(x.device)
                    
                    correlations.append(corr)
                
                batch_correlations.append(torch.stack(correlations))
            
            all_correlations.append(torch.stack(batch_correlations))
        
        # Stack all correlations into the final shape [batch_size, num_templates, signal_length]
        correlation_maps = torch.stack(all_correlations, dim=1)
        
        # Apply weights using broadcasting instead of in-place operations
        weights = F.softmax(self.template_weights, dim=0)
        weighted_maps = correlation_maps * weights.view(1, -1, 1)
        
        # Sum across template dimension
        return torch.sum(weighted_maps, dim=1)

class SpectralAnalysisModule(nn.Module):
    """Module for frequency domain analysis"""
    def __init__(self, signal_length=2048, out_features=64):
        super().__init__()
        self.signal_length = signal_length
        
        # Network to process FFT features
        self.spectral_fc = nn.Sequential(
            nn.Linear(signal_length//2 + 1, 256),  # Real and imaginary parts
            nn.ReLU(),
            nn.Linear(256, out_features),
            nn.ReLU(),
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute FFT for frequency analysis
        x_fft = torch.fft.rfft(x, dim=1)
        
        # Compute magnitude spectrum (absolute value of complex FFT)
        x_fft_mag = torch.abs(x_fft)
        
        # Process FFT features
        spectral_features = self.spectral_fc(x_fft_mag)
        
        return spectral_features  # [batch_size, out_features]

class GradientRefinementModule(nn.Module):
    def __init__(self, signal_length=2048, num_iterations=3, step_size=0.001):
        super().__init__()
        self.signal_length = signal_length
        self.num_iterations = num_iterations
        self.step_size = step_size
        # Step size based on quarter of minimum peak width (10/4 = 2.5 samples)
        self.derivative_step = (10.0 / 4.0) / signal_length
    
    def _sample_signal_values(self, signals, positions):
        """Sample signal values at given positions using linear interpolation.
        
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
        """Calculate gradients and curvatures at specified positions using interpolation.
        
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
    
@register_model("pinn_peak_detector")
class PINNPeakDetector(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Signal parameters
        self.signal_length = 2048
        
        # Flatten layer (to match the SimplestPossible interface)
        self.flatten = nn.Flatten()
        
        # Early region network (focused on first peak region)
        self.early_fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Late region network (focused on second peak region)
        self.late_fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Hanning template matching
        self.template_matcher = HanningTemplateLayer(min_width=10, max_width=40, num_templates=4)
        
        # Spectral analysis
        self.spectral_module = SpectralAnalysisModule(signal_length=self.signal_length, out_features=64)
        
        # Attention for signal weighting based on expected peak regions
        # We'll create two attention masks: one for early region (0.1-0.5) and one for late region (0.6-1.8)
        time_points = torch.linspace(0, 1, self.signal_length)
        early_mask = torch.exp(-((time_points - 0.3) / 0.1)**2)
        late_mask = torch.exp(-((time_points - 1.0) / 0.3)**2)
        self.register_buffer('early_mask', early_mask)
        self.register_buffer('late_mask', late_mask)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 64, 128),  # early + late + template + spectral
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final output layer - directly output the 3 values as required
        self.output_layer = nn.Linear(64, 3)
        
        # Gradient refinement module
        self.gradient_refiner = GradientRefinementModule(
            signal_length=self.signal_length,
            num_iterations=3,
            step_size=0.001
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure we're working with flattened input as in SimplestPossible
        x = self.flatten(x)
        batch_size = x.shape[0]
        
        # Apply early and late region attention masks
        early_weighted = x * self.early_mask.repeat(batch_size, 1)
        late_weighted = x * self.late_mask.repeat(batch_size, 1)
        
        # Process through each specialized path
        early_features = self.early_fc(early_weighted)
        late_features = self.late_fc(late_weighted)
        
        # Template matching features
        template_features = self.template_matcher(x)
        template_pooled = F.adaptive_avg_pool1d(template_features.unsqueeze(1), 64).squeeze(1)
        
        # Spectral features
        spectral_features = self.spectral_module(x)
        
        # Concatenate all feature types
        combined_features = torch.cat([
            early_features,
            late_features, 
            template_pooled,
            spectral_features
        ], dim=1)
        
        # Fuse the features
        fused = self.fusion(combined_features)
        
        # Initial output layer
        initial_output = self.output_layer(fused)
        
        # Ensure the outputs follow the constraint that peak1 < midpoint < peak2
        # This uses the same soft sorting approach as in SimplestPossible
        sorted_output = initial_output + 0.1 * (torch.sort(initial_output, dim=1)[0] - initial_output)
        
        # Apply gradient refinement to find exact zero-gradient points
        refined_output = self.gradient_refiner(x, sorted_output)
        
        return refined_output