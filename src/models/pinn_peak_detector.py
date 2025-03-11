import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import SpectralBranch, GradientRefinementModule

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
            
            # Pad template to match signal length
            padded_template = F.pad(template, (0, signal_length - template.size(0)))
            
            # Compute correlation using FFT for efficiency
            x_fft = torch.fft.rfft(x)
            template_fft = torch.fft.rfft(padded_template.repeat(batch_size, 1))
            correlation = torch.fft.irfft(x_fft * torch.conj(template_fft), n=signal_length)
            
            # Apply learnable weight for this template
            weighted_correlation = correlation * self.template_weights[i]
            all_correlations.append(weighted_correlation)
            
        # Stack and sum correlations
        return torch.stack(all_correlations, dim=1).sum(dim=1)

@register_model("pinn_peak_detector")
class PINNPeakDetector(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Signal parameters
        self.signal_length = config.signal.length
        
        # Flatten layer (to match the SimplestPossible interface)
        self.flatten = nn.Flatten()
        
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
        
        # Spectral analysis using component
        self.spectral_module = SpectralBranch(
            signal_length=self.signal_length,
            out_features=64
        )
        
        # Attention for signal weighting based on expected peak regions
        # We'll create two attention masks: one for early region (0.05-0.25) and one for late region (0.3-0.9)
        time_points = torch.linspace(0, 1, self.signal_length)
        early_mask = torch.exp(-((time_points - 0.15) / 0.1)**2)  # Centered at 0.15 (middle of 0.05-0.25)
        late_mask = torch.exp(-((time_points - 0.6) / 0.3)**2)   # Centered at 0.6 (middle of 0.3-0.9)
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
        
        # Output layer
        self.output = nn.Linear(64, 3)
        
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
        
        # Spectral features using component
        spectral_features = self.spectral_module(x.unsqueeze(1))
        
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
        x = self.output(fused)
        
        # Apply gradient refinement to find exact zero-gradient points
        x = self.refine_peaks(x, x)
        
        # Apply sigmoid activation
        x = torch.sigmoid(x)
        
        return x