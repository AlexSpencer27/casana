# Dual Pathway FFT Model

> TL;DR: Like a bilingual translator, this model speaks both time and frequency domains fluently, using attention to fuse their insights into peak detection.

## Overview
The Dual Pathway FFT model is a sophisticated neural network architecture that processes signals in both time and frequency domains simultaneously. It combines multi-scale temporal processing with spectral analysis, using cross-pathway attention to effectively fuse information from both domains.

## Architecture

### Key Components
1. **Time Domain Branch**
   - Multi-scale CNN processing
   - Kernel sizes: 7, 15, 31
   - 16 channels per kernel
   - Max pooling for feature reduction
   - Additional convolution layers (9x1 and 5x1)
   - Adaptive pooling to fixed size

2. **Frequency Domain Branch**
   - Spectral analysis using FFT
   - Windowed FFT processing
   - Separate processing of real and imaginary components
   - Feature extraction through FC layers
   - Output dimension: 64

3. **Cross-Pathway Attention**
   - Learns attention weights for each domain
   - Two-layer MLP for attention calculation
   - Softmax activation for normalized weights
   - Weighted combination of features

4. **Feature Processing Pipeline**
   - Skip connection MLP block
   - Dropout regularization (0.3)
   - Peak ordering layer
   - Final output layer (3 values)

### Data Flow
1. Input signal splits into two paths:
   - Time domain: Multi-scale CNN → Convolution blocks → Adaptive pooling
   - Frequency domain: FFT → Feature extraction → FC layers
2. Features from both paths are flattened
3. Cross-pathway attention fuses the features
4. Skip connection MLP processes fused features
5. Peak ordering ensures physical constraints

## Technical Details

### Model Parameters
- Input: 1D signal
- Output: 3 values (peak1, midpoint, peak2)
- Time branch: 64 * 16 features
- Frequency branch: 64 features
- Hidden dimensions: 256, 64
- Dropout rate: 0.3

### Key Features
- Dual domain processing
- Multi-scale temporal analysis
- Spectral feature extraction
- Cross-pathway attention
- Skip connections for gradient flow
- Peak ordering constraint

## Implementation Notes
- Uses PyTorch's nn.Module
- Inherits from BaseModel
- Implements gradient refinement capability
- Peak ordering ensures peak1 < midpoint < peak2
- Efficient FFT processing with windowing

## Advantages
- Captures both temporal and spectral information
- Multi-scale processing for robust feature extraction
- Attention mechanism for optimal feature fusion
- Skip connections improve training stability
- Physically meaningful output constraints

## Use Cases
- Peak detection in time series data
- Signal processing applications
- Feature extraction from temporal data
- Pattern recognition in sequential data
- Time-frequency analysis 