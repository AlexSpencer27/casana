import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel

@register_model("dual_pathway_fft")
class DualPathwayFFT(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Time domain path with multi-scale kernels
        self.conv1_small = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv1_medium = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        self.conv1_large = nn.Conv1d(1, 16, kernel_size=31, padding=15)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(48, 64, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        
        # Frequency domain path with band selection
        self.fft_bn = nn.BatchNorm1d(2)  # Normalize FFT components
        
        # Two separate pathways for low and high frequency bands
        self.freq_low_conv = nn.Conv1d(2, 32, kernel_size=5, padding=2)
        self.freq_high_conv = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.freq_pool = nn.MaxPool1d(2)
        
        self.freq_conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.freq_pool2 = nn.MaxPool1d(2)
        
        # Adaptive pooling to fix output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(64 * 16 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with skip connection
        self.fc1 = nn.Linear(64 * 16 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_skip = nn.Linear(64 * 16 * 2, 64)  # Skip connection
        self.output = nn.Linear(64, 3)
        
        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Multi-scale time domain processing
        t_small = self.conv1_small(x)
        t_medium = self.conv1_medium(x)
        t_large = self.conv1_large(x)
        
        # Concatenate multi-scale features
        time_features = torch.cat([t_small, t_medium, t_large], dim=1)
        time_features = self.pool1(time_features)
        
        time_features = F.relu(self.conv2(time_features))
        time_features = self.pool2(time_features)
        
        time_features = F.relu(self.conv3(time_features))
        time_features = self.adaptive_pool(time_features)
        
        # Frequency domain processing with segmented FFT
        x_squeezed = x.squeeze(1)
        
        # Apply windowed FFT for better frequency localization
        window_size = 256
        stride = 128
        
        # Process signal in overlapping windows for FFT
        windows = x_squeezed.unfold(1, window_size, stride)
        windows = windows * torch.hann_window(window_size, device=x.device)
        
        # Apply FFT to each window and take average
        fft_windows = torch.fft.rfft(windows, dim=2)
        fft_features = torch.mean(fft_windows, dim=1)
        
        # Split into real and imaginary parts and stack as channels
        real_part = fft_features.real.unsqueeze(1)
        imag_part = fft_features.imag.unsqueeze(1)
        fft_features = torch.cat([real_part, imag_part], dim=1)
        fft_features = self.fft_bn(fft_features)
        
        # Split into frequency bands and process separately
        low_freq = self.freq_low_conv(fft_features[:, :, :fft_features.shape[2]//4])
        high_freq = self.freq_high_conv(fft_features)
        
        # Resize high_freq to match low_freq size if needed
        if high_freq.size(2) != low_freq.size(2):
            high_freq = F.interpolate(high_freq, size=low_freq.size(2))
        
        # Combine frequency bands
        freq_features = torch.cat([low_freq, high_freq], dim=1)
        freq_features = self.freq_pool(freq_features)
        
        freq_features = F.relu(self.freq_conv2(freq_features))
        freq_features = self.freq_pool2(freq_features)
        
        freq_features = self.adaptive_pool(freq_features)
        
        # Flatten features
        time_features_flat = time_features.view(batch_size, -1)
        freq_features_flat = freq_features.view(batch_size, -1)
        
        # Combine features with attention mechanism
        combined_features = torch.cat([time_features_flat, freq_features_flat], dim=1)
        attention_weights = self.attention(combined_features)
        
        # Apply attention weights
        time_weighted = time_features_flat * attention_weights[:, 0].unsqueeze(1)
        freq_weighted = freq_features_flat * attention_weights[:, 1].unsqueeze(1)
        
        # Recombine weighted features
        weighted_features = torch.cat([time_weighted, freq_weighted], dim=1)
        
        # Main path through FC layers
        x = F.relu(self.bn1(self.fc1(weighted_features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Skip connection
        skip = F.relu(self.fc_skip(weighted_features))
        
        # Combine main path and skip connection
        x = x + skip
        x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        
        # Ensure peak1 < midpoint < peak2 using soft constraints
        # Sort the outputs but maintain gradient flow with a differentiable approach
        # This helps enforce peak1 < midpoint < peak2
        sorted_x = x + 0.1 * (torch.sort(x, dim=1)[0] - x)
        
        return sorted_x