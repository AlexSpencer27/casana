# Physics-Informed Neural Network Peak Detector

TL;DR: Physics-informed network leveraging spectral analysis and gradient constraints

## Overview
A Physics-Informed Neural Network (PINN) that incorporates domain knowledge through Hanning template matching, spectral analysis, and gradient-based refinement. Specializes in early and late region peak detection.

## Architecture

### Core Components
1. Region-Specific Networks
   - Early region network (first peak)
   - Late region network (second peak)
   - Attention masks for region focus

2. Hanning Template Matching
   - Multiple template widths (10-40)
   - Learnable template weights
   - FFT-based efficient correlation

3. Spectral Analysis
   - Frequency domain processing
   - Feature extraction from spectra
   - Integration with time domain features

4. Gradient Refinement
   - Physics-informed peak refinement
   - Position fine-tuning
   - Signal consistency checks

## Technical Details

### Input/Output Specifications
- Input: Signal tensor of shape `[batch_size, signal_length]`
- Output: 3-dimensional prediction tensor (sigmoid activated)

### Key Parameters
- Early region center: 0.15 (relative time)
- Late region center: 0.6 (relative time)
- Template widths: 10-40 samples
- Hidden dimensions: 256, 64
- Dropout rate: 0.2 (early/late), 0.3 (fusion)

## Implementation Notes

### Dependencies
- PyTorch
- Custom components:
  - SpectralBranch
  - GradientRefinementModule
  - HanningTemplateLayer

### Integration Guidelines
1. Signal should be normalized to [0,1] range
2. Time axis should be scaled to [0,1]
3. Early/late region masks are automatically applied
4. Gradient refinement is applied post-prediction

## Advantages
- Strong physics-based priors
- Region-specific processing
- Efficient template matching via FFT
- Gradient-based refinement
- High precision in peak localization 