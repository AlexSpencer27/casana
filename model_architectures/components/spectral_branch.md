# Spectral Branch Component

TL;DR: Frequency-domain analysis for enhanced peak discrimination

## Overview
A specialized component for processing signals in the frequency domain using windowed FFT analysis. Provides flexible spectral feature extraction with various processing modes.

## Architecture

### Core Components
1. FFT Processing
   - Windowed FFT analysis
   - Optional Hann window
   - Support for magnitude, phase, or complex processing

2. Feature Extraction
   - Adaptive window handling
   - Configurable window size and stride
   - Window averaging for long signals

3. Neural Network
   - Two-layer MLP
   - Dropout regularization
   - ReLU activations

## Technical Details

### Input/Output Specifications
- Input: Signal tensor of shape `[batch_size, channels, signal_length]`
- Output: Feature tensor of shape `[batch_size, out_features]`

### Key Parameters
- Window size: 512 (default)
- Stride: 256 (default)
- Output features: 64 (default)
- Hidden dimension: 128
- Dropout rate: 0.3
- Processing modes: magnitude, phase, complex

## Implementation Notes

### Dependencies
- PyTorch
- torch.fft module

### Integration Guidelines
1. Choose appropriate window size for signal length
2. Select processing mode based on needs:
   - 'magnitude': Spectral power features
   - 'phase': Phase information
   - 'complex': Both real and imaginary parts
3. Enable/disable windowing based on signal characteristics

## Advantages
- Flexible spectral processing
- Efficient windowed analysis
- Multiple processing modes
- Automatic dimension handling
- Memory-efficient implementation

## Use Cases
- Spectral analysis
- Frequency feature extraction
- Signal decomposition
- Pattern recognition
- Time-frequency analysis 