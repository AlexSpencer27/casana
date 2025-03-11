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
        
        # Early region network (focused on first peak region)
        self.early_fc = nn.Sequential(
            nn.Linear(config.signal.length, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Late region network (focused on second peak region)
        self.late_fc = nn.Sequential(
            nn.Linear(config.signal.length, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Hanning template matching
        self.template_matcher = HanningTemplateLayer(min_width=10, max_width=40, num_templates=4)
        
        # Spectral analysis branch
        self.spectral_module = SpectralBranch(
            signal_length=self.signal_length,
            out_features=64
        )
        
        # Create attention masks for expected peak regions
        time_points = torch.linspace(0, 1, self.signal_length)
        early_mask = torch.exp(-((time_points - 0.15) / 0.1)**2)  # Centered at 0.15
        late_mask = torch.exp(-((time_points - 0.6) / 0.3)**2)    # Centered at 0.6
        self.register_buffer('early_mask', early_mask)
        self.register_buffer('late_mask', late_mask)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4, 128),  # early + late + template + spectral
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Linear(64, 3)
        
        # Gradient refinement module
        self.gradient_refiner = GradientRefinementModule(signal_length=self.signal_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Process early and late regions with attention masks
        early_weighted = x_flat * self.early_mask
        late_weighted = x_flat * self.late_mask
        
        # Extract features from each pathway
        early_features = self.early_fc(early_weighted)
        late_features = self.late_fc(late_weighted)
        
        # Template matching features
        template_features = self.template_matcher(x)
        template_pooled = F.adaptive_avg_pool1d(template_features.unsqueeze(1), 64).squeeze(1)
        
        # Spectral features
        spectral_features = self.spectral_module(x.unsqueeze(1))
        
        # Combine all feature types
        combined_features = torch.cat([
            early_features,
            late_features, 
            template_pooled,
            spectral_features
        ], dim=1)
        
        # Fuse features and generate initial prediction
        fused = self.fusion(combined_features)
        output = self.output(fused)
        
        # Apply gradient-based refinement
        refined_output = self.gradient_refiner(output, x)
        
        return torch.sigmoid(refined_output)


class HanningTemplateLayer(nn.Module):
    """Custom layer for template matching with Hanning windows of different widths"""
    def __init__(self, min_width=10, max_width=40, num_templates=4):
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.num_templates = num_templates
        
        # Generate template widths
        self.widths = torch.linspace(min_width, max_width, num_templates, dtype=torch.int)
        self.template_weights = nn.Parameter(torch.ones(num_templates))
        
    def forward(self, x):
        # Handle both 2D and 3D input tensors
        if x.dim() == 3:  # [batch_size, channels, signal_length]
            x = x.squeeze(1)  # Remove channel dimension if present
        
        batch_size, signal_length = x.shape
        all_correlations = []
        
        for i, width in enumerate(self.widths):
            width = width.item()
            template = torch.from_numpy(np.hanning(width * 2)).float().to(x.device)
            padded_template = F.pad(template, (0, signal_length - template.size(0)))
            
            # Compute correlation using FFT for efficiency
            x_fft = torch.fft.rfft(x)
            template_fft = torch.fft.rfft(padded_template.repeat(batch_size, 1))
            correlation = torch.fft.irfft(x_fft * torch.conj(template_fft), n=signal_length)
            
            weighted_correlation = correlation * self.template_weights[i]
            all_correlations.append(weighted_correlation)
            
        return torch.stack(all_correlations, dim=1).sum(dim=1)