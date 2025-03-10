# Common Components

## Overview
The Common components module provides essential utilities and shared components used across different models. It includes specialized layers for peak ordering and adaptive feature pooling, ensuring consistent behavior and efficient processing.

## Architecture

### Key Components
1. **PeakOrderingLayer**
   - Position normalization
   - Order constraints
   - Minimum separation
   - Soft constraints
   - Differentiable operations

2. **AdaptiveFeaturePooling**
   - Flexible input sizes
   - Multiple pooling types
   - Fixed output size
   - Efficient computation
   - Clean interface

### Data Flow
1. **PeakOrderingLayer**
   - Input positions → Clamp to [0,1]
   - Split into peak1, midpoint, peak2
   - Apply minimum separation
   - Recompute midpoint
   - Concatenate results

2. **AdaptiveFeaturePooling**
   - Input features → Pooling operation
   - Resize to fixed output
   - Return processed features

## Technical Details

### Module Parameters
#### PeakOrderingLayer
- Softness: 0.1 (default)
- Minimum separation: 0.1 (default)
- Input shape: [batch_size, 3]
- Output shape: [batch_size, 3]

#### AdaptiveFeaturePooling
- Output size: 16 (default)
- Pooling type: 'avg' or 'max'
- Input shape: [batch_size, channels, length]
- Output shape: [batch_size, channels, output_size]

### Key Features
- Position constraints
- Adaptive pooling
- Differentiable operations
- Efficient computation
- Flexible configuration

## Implementation Notes
- Uses PyTorch's nn.Module
- Efficient tensor operations
- Clean interface
- Memory efficient
- Robust to input variations

## Advantages
- Ensures physical constraints
- Handles variable input sizes
- Efficient computation
- Easy to integrate
- Memory efficient

## Use Cases
- Peak position normalization
- Feature map resizing
- Signal processing
- Pattern detection
- Feature extraction 