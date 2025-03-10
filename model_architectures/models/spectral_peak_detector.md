# Spectral Peak Detector Model

> TL;DR: The Swiss Army knife of peak detection, this model slices through signals in multiple ways - time, frequency, magnitude, and phase - to find those elusive peaks.

## Overview
The Spectral Peak Detector model is a specialized architecture that combines time-domain processing with advanced spectral analysis. It processes signals through multiple pathways, including magnitude and phase analysis of the STFT, to achieve robust peak detection.

## Architecture

### Key Components
1. **Time Domain Branch**
   - Multi-scale CNN processing
   - Kernel sizes: 15, 7
   - 16 channels per kernel
   - Max pooling (size 4)
   - Feature dimension: 32 * (2048/16)

2. **Spectral Branch**
   - FFT-based spectral analysis
   - Separate processing of real/imaginary components
   - Feature dimension: 64

3. **STFT Processing**
   - Short-time Fourier transform
   - N_FFT: 256
   - Hop length: 128
   - Window length: 256
   - Hann windowing

4. **Magnitude Processing**
   - Two convolution layers (5x1, 3x1)
   - 32 output channels
   - Max pooling (size 2)
   - Feature dimension: 32 * ((2048/128)/2)

5. **Phase Processing**
   - Single convolution layer (5x1)
   - 16 output channels
   - Max pooling (size 2)
   - Feature dimension: 16 * ((2048/128)/2)

6. **Feature Fusion**
   - Skip connection MLP
   - Hidden dimension: 256
   - Output dimension: 128
   - Dropout rate: 0.3

### Data Flow
1. Input signal processes through multiple paths:
   - Time domain: Multi-scale CNN → Pooling
   - Spectral: FFT → Feature extraction
   - STFT: Magnitude and phase processing
2. All features are concatenated
3. Skip connection MLP fuses features
4. Final layers process fused features
5. Peak ordering ensures physical constraints

## Technical Details

### Model Parameters
- Input: 1D signal
- Output: 3 values (peak1, midpoint, peak2)
- Feature dimensions:
  - Time: 32 * (2048/16)
  - Spectral: 64
  - Magnitude: 32 * ((2048/128)/2)
  - Phase: 16 * ((2048/128)/2)
- Hidden dimensions: 256, 128, 64
- Dropout rate: 0.3

### Key Features
- Multi-pathway processing
- Advanced spectral analysis
- Magnitude and phase analysis
- Skip connections
- Peak ordering constraint
- Adaptive feature fusion

## Implementation Notes
- Uses PyTorch's nn.Module
- Inherits from BaseModel
- Implements gradient refinement capability
- Peak ordering ensures peak1 < midpoint < peak2
- Efficient STFT processing with windowing
- Dynamic feature dimension handling

## Advantages
- Comprehensive signal analysis
- Robust to noise through multiple pathways
- Captures both magnitude and phase information
- Skip connections improve training stability
- Physically meaningful output constraints

## Use Cases
- Peak detection in time series data
- Signal processing applications
- Feature extraction from temporal data
- Pattern recognition in sequential data
- Spectral analysis applications 