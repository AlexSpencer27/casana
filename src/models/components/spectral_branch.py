"""
Spectral branch components for processing signals in frequency domain using FFT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralBranch(nn.Module):
    def __init__(
        self,
        signal_length,
        out_features=64,
        window_size=512,
        stride=256,
        use_window=True,
        mode='magnitude'
    ):
        """
        Spectral analysis branch that processes input signal using windowed FFT.
        
        Args:
            signal_length: Length of input signal
            out_features: Number of output features (default: 64)
            window_size: Size of FFT window (default: 512)
            stride: Stride between windows (default: 256)
            use_window: Whether to apply Hann window (default: True)
            mode: How to process FFT results ('magnitude', 'phase', or 'complex') (default: 'magnitude')
        """
        super().__init__()
        self.signal_length = signal_length
        self.window_size = min(window_size, signal_length)
        self.stride = min(stride, signal_length)
        self.use_window = use_window
        self.mode = mode
        
        # Create Hann window if enabled
        if use_window:
            self.register_buffer('window', torch.hann_window(self.window_size))
        
        # Calculate FFT feature size based on mode
        if mode == 'complex':
            fft_size = self.window_size + 2  # Real and imaginary parts
        else:
            fft_size = self.window_size // 2 + 1  # Only magnitude or phase
            
        # Fully connected layers for spectral feature processing
        self.spectral_fc = nn.Sequential(
            nn.Linear(fft_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_features),
            nn.ReLU()
        )
    
    def _process_fft(self, fft_result):
        """Process FFT results based on specified mode."""
        if self.mode == 'magnitude':
            return torch.abs(fft_result)
        elif self.mode == 'phase':
            return torch.angle(fft_result)
        else:  # complex
            return torch.cat([torch.real(fft_result), torch.imag(fft_result)], dim=-1)
    
    def forward(self, x):
        """
        Forward pass for the spectral branch.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, signal_length]
            
        Returns:
            Tensor of shape [batch_size, out_features]
        """
        batch_size = x.shape[0]
        x_squeezed = x.squeeze(1) if x.dim() > 2 else x
        
        if self.window_size < self.signal_length:
            # Apply windowed FFT for better frequency localization
            # Unfold along the signal dimension (last dimension)
            windows = x_squeezed.unfold(-1, self.window_size, self.stride)  # [batch_size, num_windows, window_size]
            
            # Apply Hann window if enabled
            if self.use_window:
                windows = windows * self.window  # Broadcasting to [batch_size, num_windows, window_size]
            
            # Reshape windows to [batch_size * num_windows, window_size] for batch processing
            num_windows = windows.size(1)
            windows = windows.reshape(-1, self.window_size)
            
            # Apply Real FFT to each window
            fft_result = torch.fft.rfft(windows)  # [batch_size * num_windows, fft_size]
            
            # Process FFT results based on specified mode
            processed_fft = self._process_fft(fft_result)  # [batch_size * num_windows, fft_size]
            
            # Reshape back to [batch_size, num_windows, fft_size]
            processed_fft = processed_fft.view(batch_size, num_windows, -1)
            
            # Average across windows to get [batch_size, fft_size]
            fft_features = torch.mean(processed_fft, dim=1)
        else:
            # Apply Real FFT to entire signal
            fft_result = torch.fft.rfft(x_squeezed)  # [batch_size, fft_size]
            
            # Process FFT results
            fft_features = self._process_fft(fft_result)  # [batch_size, fft_size]
        
        # Get the expected input size from the first layer of spectral_fc
        expected_input_size = self.spectral_fc[0].weight.shape[1]
        actual_input_size = fft_features.shape[1]
        
        # If dimensions don't match, reshape or pad as needed
        if actual_input_size != expected_input_size:
            if actual_input_size < expected_input_size:
                # Pad with zeros if actual is smaller
                padding = torch.zeros(batch_size, expected_input_size - actual_input_size, device=fft_features.device)
                fft_features = torch.cat([fft_features, padding], dim=1)  # [batch_size, expected_input_size]
            else:
                # Truncate if actual is larger
                fft_features = fft_features[:, :expected_input_size]  # [batch_size, expected_input_size]
        
        # Process through fully connected layers
        return self.spectral_fc(fft_features)  # [batch_size, out_features] 