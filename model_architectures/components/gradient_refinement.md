# Gradient Refinement

TL;DR: "Trust me, that peak is *slightly* to the left... 🎯"

## Overview
A physics-informed module that refines peak predictions using signal gradients and curvature information. Uses iterative optimization to find the most precise peak locations.

## Architecture

### Core Components
1. Position Sampling
   - Linear interpolation
   - Sub-pixel precision
   - Boundary handling

2. Gradient Analysis
   - Central difference gradients
   - Curvature computation
   - Finite difference approximations

3. Iterative Refinement
   - Newton's method optimization
   - Adaptive step sizing
   - Position clamping

## Technical Details

### Input/Output Specifications
- Input: 
  - Initial predictions: `[batch_size, 3]`
  - Signals: `[batch_size, signal_length]`
- Output: Refined predictions `[batch_size, 3]`

### Key Parameters
- Base step size: 0.002
- Maximum iterations: 7
- Epsilon (finite difference): 0.01
- Position range: [0, 1]

## Implementation Notes

### Dependencies
- PyTorch
- No external components required

### Integration Guidelines
1. Signals should be normalized
2. Initial predictions should be in [0, 1] range
3. Can handle both 2D and 3D input signals
4. Automatically handles batch processing

## Advantages
- Physics-informed refinement
- Sub-pixel precision
- Numerically stable
- Batch-optimized implementation
- Automatic boundary handling

## Use Cases
- Fine-tuning peak locations
- Improving model predictions
- Signal extrema detection
- High-precision localization
- Post-processing refinement 