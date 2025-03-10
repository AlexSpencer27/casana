# Spectral Branch Component

## Overview
The Spectral Branch component is a specialized module for processing signals in the frequency domain using FFT. It supports windowed FFT processing, different complex number handling modes, and efficient feature extraction from spectral representations.

## Architecture

### Key Components
1. **FFT Processing**
   - Windowed FFT support
   - Configurable window size
   - Adjustable stride
   - Hann window option
   - Real FFT computation

2. **Complex Processing Modes**
   - Magnitude spectrum
   - Separate real/imaginary
   - Complex feature extraction
   - Phase preservation
   - Magnitude-phase combination

3. **Feature Extraction**
   - Two-layer FC network
   - ReLU activation
   - Dropout regularization
   - Configurable output dimension
   - Adaptive input handling

### Data Flow
1. Input signal → Windowed FFT
2. Complex number processing
3. Feature extraction
4. Output features

## Technical Details

### Module Parameters
- Signal length: configurable (default: 2048)
- Output features: configurable (default: 64)
- Window size: adaptive (default: signal_length/4)
- Stride: adaptive (default: window_size/2)
- Use window: configurable (default: True)
- Process complex: 'magnitude', 'separate', or 'complex'

### Key Features
- Windowed FFT processing
- Multiple complex modes
- Adaptive window sizing
- Efficient feature extraction
- Flexible configuration

## Implementation Notes
- Uses PyTorch's nn.Module
- Efficient FFT computation
- Memory-efficient processing
- Robust to input variations
- Clean interface

## Advantages
- Frequency domain analysis
- Window-based processing
- Multiple complex modes
- Efficient computation
- Flexible configuration

## Use Cases
- Spectral analysis
- Frequency feature extraction
- Signal decomposition
- Pattern recognition
- Time-frequency analysis 