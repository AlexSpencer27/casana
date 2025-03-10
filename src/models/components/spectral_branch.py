"""
Spectral branch components for processing signals in frequency domain using FFT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralBranch(nn.Module):
    def __init__(
        self,
        signal_length=2048,
        out_features=64,
        window_size=None,
        stride=None,
        use_window=True,
        process_complex='magnitude'
    ):
        """
        Spectral branch for processing signals in frequency domain using Real FFT.
        
        Args:
            signal_length: Original signal length (default: 2048)
            out_features: Output feature dimension (default: 64)
            window_size: Size of sliding window for windowed FFT (default: signal_length // 4)
            stride: Stride for windowed FFT (default: window_size // 2)
            use_window: Whether to apply Hann window (default: True)
            process_complex: How to process complex values ('magnitude', 'separate', 'complex') (default: 'magnitude')
        """
        super().__init__()
        
        self.signal_length = signal_length
        
        # Calculate adaptive window_size and stride if not provided
        if window_size is None:
            self.window_size = signal_length // 4
        else:
            self.window_size = window_size
        
        if stride is None:
            self.stride = self.window_size // 2
        else:
            self.stride = stride
        
        self.use_window = use_window
        self.process_complex = process_complex
        
        # Determine input size for the FC network based on window size
        if self.window_size < self.signal_length:
            # Using windowed FFT - Real FFT has window_size//2 + 1 output size
            fft_output_size = self.window_size // 2 + 1
        else:
            # Using full signal FFT - Real FFT has signal_length//2 + 1 output size
            fft_output_size = signal_length // 2 + 1
            
        # If we're processing real and imaginary parts separately, double the size
        if process_complex == 'separate':
            fft_output_size *= 2
            
        # Network to process FFT features
        self.spectral_fc = nn.Sequential(
            nn.Linear(fft_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_features),
            nn.ReLU(),
        )
        
        # Register Hann window if using windowing
        if self.use_window:
            self.register_buffer('window', torch.hann_window(self.window_size))
            
    def _process_fft(self, fft_result):
        """
        Process FFT results based on the specified mode.
        
        Args:
            fft_result: Complex FFT result
            
        Returns:
            Processed FFT features
        """
        if self.process_complex == 'magnitude':
            # Return magnitude spectrum (absolute value)
            return torch.abs(fft_result)
        
        elif self.process_complex == 'separate':
            # Return real and imaginary parts concatenated
            real_part = fft_result.real
            imag_part = fft_result.imag
            return torch.cat([real_part, imag_part], dim=1)
            
        elif self.process_complex == 'complex':
            # For complex, we have a more sophisticated way to handle
            # This is a simplified approach - in practice you might use complex-valued networks
            magnitudes = torch.abs(fft_result)
            phases = torch.angle(fft_result)
            # Convert to feature representation that preserves complex info
            features = magnitudes * torch.cos(phases)
            return features
    
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
            windows = x_squeezed.unfold(1, self.window_size, self.stride)
            
            # Apply Hann window if enabled
            if self.use_window:
                windows = windows * self.window
            
            # Apply Real FFT to each window
            fft_windows = torch.fft.rfft(windows, dim=2)
            
            # Process FFT results based on specified mode
            processed_fft = self._process_fft(fft_windows)
            
            # Average across windows
            fft_features = torch.mean(processed_fft, dim=1)
        else:
            # Apply Real FFT to entire signal
            fft_result = torch.fft.rfft(x_squeezed)
            
            # Process FFT results
            fft_features = self._process_fft(fft_result)
        
        # Get the expected input size from the first layer of spectral_fc
        expected_input_size = self.spectral_fc[0].weight.shape[1]
        actual_input_size = fft_features.shape[1]
        
        # If dimensions don't match, reshape or pad as needed
        # This handles the case where the FFT output size doesn't match the expected input size
        # for the linear layer, which can happen when using windowed FFT with different window sizes
        # or when processing real and imaginary parts separately
        if actual_input_size != expected_input_size:
            if actual_input_size < expected_input_size:
                # Pad with zeros if actual is smaller
                padding = torch.zeros(batch_size, expected_input_size - actual_input_size, device=fft_features.device)
                fft_features = torch.cat([fft_features, padding], dim=1)
            else:
                # Truncate if actual is larger
                fft_features = fft_features[:, :expected_input_size]
        
        # Process FFT features through FC network
        spectral_features = self.spectral_fc(fft_features)
        
        return spectral_features 