import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import SpectralBranch, GradientRefinementModule

@register_model("pinn_peak_detector")
class PINNPeakDetector(BaseModel):
    """Physics-informed neural network that incorporates domain knowledge through 
    Hanning template matching, spectral analysis, and gradient-based refinement."""
    
    def __init__(self) -> None:
        super().__init__()
        
        # Signal parameters
        self.signal_length = config.signal.length
        
        # Noise filtering branch
        self.noise_filter = NoiseFilterBranch(
            signal_length=self.signal_length,
            min_freq=1,  # Match signal generator noise range
            max_freq=10,
            out_features=128  # Increased features for better noise characterization
        )
        
        # Template matching with explicit peak width range
        self.template_matcher = HanningTemplateLayer(
            min_width=10, 
            max_width=40,
            num_templates=32  # Increased templates for finer-grained matching
        )
        
        # Create region masks and time points
        time_points = torch.linspace(0, 1, self.signal_length)
        # Wider masks to allow more flexibility
        early_mask = ((time_points >= 0.025) & (time_points <= 0.3)).float()
        late_mask = ((time_points >= 0.25) & (time_points <= 0.95)).float()
        self.register_buffer('early_mask', early_mask)
        self.register_buffer('late_mask', late_mask)
        self.register_buffer('time_points', time_points)
        
        # Enhanced peak detection network with residual connections
        input_size = self.signal_length + 128 + self.signal_length  # Cleaned signal + spectral + template features
        
        # Initial projection layer - reduced capacity
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)  # Increased dropout
        )
        
        # Residual blocks - reduced number and capacity
        self.res_blocks = nn.ModuleList([
            self._make_res_block(256) for _ in range(2)  # Reduced from 4 to 2
        ])
        
        # Output projection - simpler architecture
        self.output_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Direct position prediction heads
        self.peak1_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # Direct position prediction
            nn.Sigmoid()  # Constrain to [0,1]
        )
        
        self.peak2_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # Direct position prediction
            nn.Sigmoid()  # Constrain to [0,1]
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Gradient refinement module with increased step size
        self.gradient_refiner = GradientRefinementModule(
            signal_length=self.signal_length,
            base_step_size=0.005,  # Increased from 0.002
            max_iterations=10  # Increased from 7
        )
    
    def _make_res_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 1. Filter out noise frequencies and get noise features
        noise_features, cleaned_signal = self.noise_filter(x.unsqueeze(1))
        cleaned_flat = cleaned_signal.squeeze(1).view(batch_size, -1)
        
        # 2. Find potential peaks using template matching on cleaned signal
        peak_responses = self.template_matcher(cleaned_signal)
        
        # 3. Combine all features
        noise_features = noise_features.squeeze(1)
        features = torch.cat([cleaned_flat, noise_features, peak_responses], dim=1)
        
        # Apply enhanced network with residual connections
        x_features = self.input_projection(features)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            residual = x_features
            x_features = res_block(x_features)
            x_features = x_features + residual  # Residual connection
        
        # Final projection
        shared_features = self.output_projection(x_features)
        
        # Direct position predictions
        peak1 = self.peak1_head(shared_features)
        peak2 = self.peak2_head(shared_features)
        
        # Ensure peak2 is after peak1
        peak1, peak2 = torch.min(torch.cat([peak1, peak2], dim=1), dim=1, keepdim=True)[0], \
                       torch.max(torch.cat([peak1, peak2], dim=1), dim=1, keepdim=True)[0]
        
        # Calculate midpoint
        midpoint = (peak1 + peak2) / 2
        
        # Combine outputs
        output = torch.cat([peak1, midpoint, peak2], dim=1)
        
        # Apply gradient refinement for final adjustment
        refined_output = self.gradient_refiner(output, x)
        
        return refined_output


class HanningTemplateLayer(nn.Module):
    """Custom layer for template matching with Hanning windows of different widths.
    
    Accepts inputs of shape:
        - [batch_size, signal_length] or
        - [batch_size, channels, signal_length] or
        - [batch_size, 1, 1, signal_length]
    """
    def __init__(self, min_width=10, max_width=40, num_templates=8):
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.num_templates = num_templates
        
        # Generate template widths
        self.widths = torch.linspace(min_width, max_width, num_templates, dtype=torch.int)
        self.template_weights = nn.Parameter(torch.ones(num_templates))
        
    def forward(self, x):
        # Handle input tensors of various shapes
        if x.dim() == 4:  # [batch_size, 1, 1, signal_length]
            x = x.squeeze(2).squeeze(1)
        elif x.dim() == 3:  # [batch_size, channels, signal_length]
            x = x.squeeze(1)
        elif x.dim() != 2:  # Must be 2D, 3D, or 4D
            raise ValueError(f"Input must be 2D, 3D, or 4D tensor, got shape {x.shape}")
        
        batch_size, signal_length = x.shape
        all_correlations = []
        
        for i, width in enumerate(self.widths):
            width = width.item()
            template = torch.from_numpy(np.hanning(width * 2)).float().to(x.device)
            padded_template = F.pad(template, (0, signal_length - template.size(0)))
            
            # Compute correlation using FFT for efficiency
            x_fft = torch.fft.rfft(x, n=signal_length)  # Explicit length
            template_fft = torch.fft.rfft(padded_template.repeat(batch_size, 1), n=signal_length)  # Explicit length
            correlation = torch.fft.irfft(x_fft * torch.conj(template_fft), n=signal_length)  # Match input length
            
            weighted_correlation = correlation * self.template_weights[i]
            all_correlations.append(weighted_correlation)
            
        # Stack correlations and sum along template dimension to get [batch_size, signal_length]
        stacked = torch.stack(all_correlations, dim=1)  # [batch_size, num_templates, signal_length]
        return stacked.sum(dim=1)  # [batch_size, signal_length]

class NoiseFilterBranch(nn.Module):
    """Branch that filters low-frequency noise using a high-pass filter"""
    def __init__(self, signal_length, min_freq, max_freq, out_features):
        super().__init__()
        self.signal_length = signal_length
        
        # Create frequency basis accounting for sampling rate
        sampling_rate = config.signal.sampling_rate
        freqs = torch.fft.rfftfreq(signal_length, d=1.0/sampling_rate)
        self.register_buffer('freqs', freqs)
        
        # Create high-pass filter mask (cutoff at 15 Hz, with smooth transition)
        cutoff_freq = 15.0  # Hz
        transition_width = 5.0  # Hz
        
        # Smooth transition using sigmoid
        filter_mask = 1.0 / (1.0 + torch.exp(-(freqs - cutoff_freq) / (transition_width/4)))
        self.register_buffer('filter_mask', filter_mask)
        
        # Feature extraction from frequency magnitudes
        self.freq_features = nn.Sequential(
            nn.Linear(signal_length // 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )
    
    def forward(self, x):
        # Convert to frequency domain
        x = x.squeeze(1)  # Remove channel dimension
        fft = torch.fft.rfft(x, n=self.signal_length)  # Explicit length
        magnitudes = torch.abs(fft)
        
        # Apply high-pass filter
        cleaned_fft = fft * self.filter_mask.unsqueeze(0)
        cleaned_signal = torch.fft.irfft(cleaned_fft, n=self.signal_length)  # Match input length
        
        # Extract features from frequency magnitudes for downstream use
        freq_features = self.freq_features(magnitudes)  # [batch_size, out_features]
        
        # Return freq_features as 2D and cleaned_signal as 3D
        return freq_features, cleaned_signal.unsqueeze(1)