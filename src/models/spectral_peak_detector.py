import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel

@register_model("spectral_peak_detector")
class SpectralPeakDetector(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # STFT parameters
        self.n_fft = 256
        self.hop_length = 128
        self.window_length = 256
        
        # Frequency domain processing
        self.freq_conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2))
        self.freq_conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.freq_pool = nn.MaxPool2d(2)
        
        # Secondary processing on magnitude spectrum
        self.mag_conv1 = nn.Conv1d(129, 32, kernel_size=5, padding=2)
        self.mag_conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.mag_pool = nn.MaxPool1d(2)
        
        # Phase processing
        self.phase_conv = nn.Conv1d(129, 16, kernel_size=5, padding=2)
        self.phase_pool = nn.MaxPool1d(2)
        
        # Time domain processing
        self.time_conv1 = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        self.time_conv2 = nn.Conv1d(16, 16, kernel_size=7, padding=3)
        self.time_pool = nn.MaxPool1d(4)
        
        # Calculate actual feature dimensions based on STFT and pooling
        freq_out_size = 32 * ((self.n_fft//2+1) // 4) * ((2048 // self.hop_length) // 4)
        mag_out_size = 32 * ((2048 // self.hop_length) // 2)
        phase_out_size = 16 * ((2048 // self.hop_length) // 2)
        time_out_size = 16 * (2048 // 4)
        
        # Feature fusion with correctly calculated dimensions
        self.fusion_fc = nn.Linear(
            freq_out_size + mag_out_size + phase_out_size + time_out_size,
            256
        )
        
        # Output layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 3)
        
        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Time domain processing path
        time_features = F.relu(self.time_conv1(x))
        time_features = F.relu(self.time_conv2(time_features))
        time_features = self.time_pool(time_features)
        
        # STFT computation
        x_flat = x.reshape(-1, x.size(-1))
        stft = torch.stft(
            x_flat, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=torch.hann_window(self.window_length).to(x.device),
            return_complex=True,
            normalized=True
        )
        
        # Extract magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Reshape for processing
        magnitude = magnitude.reshape(batch_size, 1, stft.size(1), stft.size(2))
        
        # 2D convolution on spectrogram
        freq_features = F.relu(self.freq_conv1(magnitude))
        freq_features = self.freq_pool(freq_features)
        freq_features = F.relu(self.freq_conv2(freq_features))
        freq_features = self.freq_pool(freq_features)
        
        # 1D processing on magnitude slices
        mag_features = magnitude.squeeze(1)
        mag_features = F.relu(self.mag_conv1(mag_features))
        mag_features = F.relu(self.mag_conv2(mag_features))
        mag_features = self.mag_pool(mag_features)
        
        # Phase processing
        phase = phase.reshape(batch_size, stft.size(1), stft.size(2))
        phase_features = F.relu(self.phase_conv(phase))
        phase_features = self.phase_pool(phase_features)
        
        # Flatten all feature paths
        freq_features = freq_features.view(batch_size, -1)
        mag_features = mag_features.view(batch_size, -1)
        phase_features = phase_features.view(batch_size, -1)
        time_features = time_features.view(batch_size, -1)
        
        # Concatenate all features
        combined = torch.cat([freq_features, mag_features, phase_features, time_features], dim=1)
        
        # Feature fusion
        x = F.relu(self.bn1(self.fusion_fc(combined)))
        x = self.dropout(x)
        
        # Output layers
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        # Ensure peak1 < midpoint < peak2 using soft constraints
        sorted_x = x + 0.1 * (torch.sort(x, dim=1)[0] - x)
        
        return sorted_x