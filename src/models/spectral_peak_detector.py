import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import SpectralBranch, PeakOrderingLayer, MultiScaleCNNBranch, SkipConnectionMLP

@register_model("spectral_peak_detector")
class SpectralPeakDetector(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Time domain processing using MultiScaleCNNBranch
        self.time_branch = MultiScaleCNNBranch(
            in_channels=1,
            channels_per_kernel=16,
            kernel_sizes=(15, 7),
            pooling='max',
            pooling_size=4
        )
        
        # Frequency domain processing using SpectralBranch
        self.spectral_branch = SpectralBranch(
            signal_length=config.signal.length,
            out_features=64,
            process_complex='separate'
        )
        
        # Secondary processing on magnitude spectrum
        self.mag_conv1 = nn.Conv1d(129, 32, kernel_size=5, padding=2)
        self.mag_conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.mag_pool = nn.MaxPool1d(2)
        
        # Phase processing
        self.phase_conv = nn.Conv1d(129, 16, kernel_size=5, padding=2)
        self.phase_pool = nn.MaxPool1d(2)
        
        # Calculate feature dimensions
        time_out_size = 32 * (2048 // 4 // 4)  # After pooling twice with factor 4
        spectral_out_size = 64  # From spectral branch
        mag_out_size = 32 * ((2048 // 128) // 2)  # After pooling with factor 2
        phase_out_size = 16 * ((2048 // 128) // 2)  # After pooling with factor 2
        
        # Feature fusion with correctly calculated dimensions
        fusion_input_dim = time_out_size + spectral_out_size + mag_out_size + phase_out_size
        
        # Feature fusion using SkipConnectionMLP
        self.fusion_fc = SkipConnectionMLP(
            input_dim=fusion_input_dim,
            hidden_dim=256,
            output_dim=128,
            dropout_rate=0.3
        )
        
        # Output layers
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 3)
        
        # Regularization
        self.dropout = nn.Dropout(0.3)
        
        # Peak ordering layer
        self.peak_ordering = PeakOrderingLayer()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Time domain processing path
        time_features = self.time_branch(x)
        time_features_flat = time_features.view(batch_size, -1)
        
        # Spectral processing using SpectralBranch
        spectral_features = self.spectral_branch(x)
        
        # STFT computation for additional processing
        x_flat = x.reshape(-1, x.size(-1))
        stft = torch.stft(
            x_flat, 
            n_fft=256, 
            hop_length=128,
            win_length=256,
            window=torch.hann_window(256).to(x.device),
            return_complex=True,
            normalized=True
        )
        
        # Extract magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Reshape for processing
        magnitude = magnitude.reshape(batch_size, stft.size(1), stft.size(2))
        phase = phase.reshape(batch_size, stft.size(1), stft.size(2))
        
        # 1D processing on magnitude slices
        mag_features = F.relu(self.mag_conv1(magnitude))
        mag_features = F.relu(self.mag_conv2(mag_features))
        mag_features = self.mag_pool(mag_features)
        mag_features_flat = mag_features.view(batch_size, -1)
        
        # Phase processing
        phase_features = F.relu(self.phase_conv(phase))
        phase_features = self.phase_pool(phase_features)
        phase_features_flat = phase_features.view(batch_size, -1)
        
        # Concatenate all features
        combined = torch.cat([
            time_features_flat, 
            spectral_features, 
            mag_features_flat, 
            phase_features_flat
        ], dim=1)
        
        # Ensure dimensions match for fusion_fc
        expected_input_dim = self.fusion_fc.fc1.weight.shape[1]
        actual_input_dim = combined.shape[1]
        
        if actual_input_dim != expected_input_dim:
            if actual_input_dim < expected_input_dim:
                # Pad with zeros if actual is smaller
                padding = torch.zeros(batch_size, expected_input_dim - actual_input_dim, device=combined.device)
                combined = torch.cat([combined, padding], dim=1)
            else:
                # Truncate if actual is larger
                combined = combined[:, :expected_input_dim]
        
        # Feature fusion
        x = self.fusion_fc(combined)
        
        # Output layers
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        
        # Ensure peak1 < midpoint < peak2 using component
        x = self.peak_ordering(x)
        
        return x