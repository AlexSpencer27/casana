# Physics-Informed Neural Network Peak Detector

TL;DR: Physics-informed network leveraging spectral analysis and gradient constraints

## Overview
A Physics-Informed Neural Network (PINN) that incorporates domain knowledge through Hanning template matching, spectral analysis, and gradient-based refinement. Specializes in noise filtering and robust peak detection.

## Architecture

### Core Components
1. Noise Filtering Branch
   - High-pass filter with smooth transition
   - Cutoff frequency at 15 Hz
   - Frequency domain feature extraction
   - Clean signal reconstruction

2. Hanning Template Matching
   - Multiple template widths (10-40)
   - Learnable template weights
   - FFT-based efficient correlation
   - 32 templates for fine-grained matching

3. Peak Detection Network
   - Residual architecture with dropout
   - Layer normalization
   - GELU activation functions
   - Direct position prediction heads

4. Gradient Refinement
   - Physics-informed peak refinement
   - Increased step size (0.005)
   - Up to 10 refinement iterations
   - Position fine-tuning

## Technical Details

### Input/Output Specifications
- Input: Signal tensor of shape `[batch_size, signal_length]`
- Output: 3-dimensional prediction tensor (sigmoid activated)
  - First peak position
  - Midpoint
  - Second peak position

### Key Parameters
- Signal length: Configurable via config
- Template widths: 10-40 samples (32 templates)
- Hidden dimensions: 256, 128, 64
- Dropout rates: 0.2 (main), 0.1 (output)
- High-pass filter cutoff: 15 Hz
- Filter transition width: 5 Hz

## Implementation Notes

### Dependencies
- PyTorch
- NumPy
- Custom components:
  - NoiseFilterBranch
  - HanningTemplateLayer
  - GradientRefinementModule

### Integration Guidelines
1. Signal should be normalized
2. Sampling rate must be specified in config
3. Expects single-channel input
4. Automatically ensures peak1 < peak2

## Advantages
- Robust noise filtering
- Efficient template matching via FFT
- Residual connections for better gradient flow
- Xavier weight initialization
- Adaptive gradient refinement
- Guaranteed peak ordering 