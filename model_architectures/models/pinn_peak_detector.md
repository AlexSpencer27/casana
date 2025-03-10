# PINN Peak Detector Model

> TL;DR: The physicist's neural network - it doesn't just learn patterns, it understands the physics behind them, using template matching and region-specific processing to find peaks with scientific precision.

## Overview
The Physics-Informed Neural Network (PINN) Peak Detector is a specialized architecture that combines neural network processing with physics-based constraints. It uses template matching, spectral analysis, and region-specific processing to achieve accurate peak detection while respecting physical constraints.

## Architecture

### Key Components
1. **Region-Specific Networks**
   - Early region network (0.05-0.25 signal range)
   - Late region network (0.3-0.9 signal range)
   - Each with 256 → 64 hidden dimensions
   - Dropout rate: 0.2

2. **Hanning Template Layer**
   - Custom layer for template matching
   - Width range: 10-40 samples
   - 4 different template widths
   - Learnable template weights
   - FFT-based correlation computation

3. **Spectral Analysis**
   - FFT-based spectral processing
   - Feature dimension: 64
   - Windowed processing
   - Magnitude spectrum analysis

4. **Attention Masks**
   - Early region mask (centered at 0.15)
   - Late region mask (centered at 0.6)
   - Gaussian weighting
   - Region-specific feature extraction

5. **Feature Fusion**
   - Concatenation of all features
   - 128 → 64 hidden dimensions
   - Dropout rate: 0.3
   - Final output layer (3 values)

### Data Flow
1. Input signal processes through multiple paths:
   - Early region: Masked input → Early network
   - Late region: Masked input → Late network
   - Template matching: Hanning templates → Correlation
   - Spectral: FFT → Feature extraction
2. All features are concatenated
3. Fusion network processes combined features
4. Peak ordering ensures physical constraints

## Technical Details

### Model Parameters
- Input: 1D signal
- Output: 3 values (peak1, midpoint, peak2)
- Feature dimensions:
  - Early region: 64
  - Late region: 64
  - Template: 64 (after pooling)
  - Spectral: 64
- Hidden dimensions: 256, 128, 64
- Dropout rates: 0.2, 0.3

### Key Features
- Physics-informed processing
- Region-specific analysis
- Template matching
- Spectral analysis
- Attention-based weighting
- Peak ordering constraint

## Implementation Notes
- Uses PyTorch's nn.Module
- Inherits from BaseModel
- Implements gradient refinement capability
- Peak ordering ensures peak1 < midpoint < peak2
- Efficient FFT-based template matching
- Custom Hanning template layer

## Advantages
- Physics-based constraints improve accuracy
- Region-specific processing for better localization
- Template matching for pattern recognition
- Spectral analysis for frequency information
- Attention masks for focused processing
- Physically meaningful output constraints

## Use Cases
- Peak detection in time series data
- Signal processing applications
- Feature extraction from temporal data
- Pattern recognition in sequential data
- Physics-constrained signal analysis 