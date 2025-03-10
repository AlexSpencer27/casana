# Multi-Scale CNN Component

## Overview
The Multi-Scale CNN component is a flexible module that processes input signals using parallel convolutional branches with different kernel sizes. This enables the capture of features at multiple temporal scales simultaneously, making it particularly effective for signal processing tasks.

## Architecture

### Key Components
1. **Parallel Convolution Branches**
   - Multiple kernel sizes
   - Configurable channels per kernel
   - Same padding for size preservation
   - ReLU activation
   - Channel concatenation

2. **Pooling Layer**
   - Configurable pooling type
   - Options: max, average, or none
   - Adjustable pooling size
   - Optional operation

3. **Activation Function**
   - Default: ReLU
   - Configurable activation
   - Applied per branch

### Data Flow
1. Input signal → Parallel convolutions
2. Activation per branch
3. Channel concatenation
4. Optional pooling
5. Output features

## Technical Details

### Module Parameters
- Input channels: configurable (default: 1)
- Channels per kernel: configurable (default: 16)
- Kernel sizes: configurable tuple (default: (7, 15, 31))
- Pooling type: 'max', 'avg', or None
- Pooling size: configurable (default: 2)
- Activation: configurable (default: ReLU)

### Key Features
- Multi-scale processing
- Parallel feature extraction
- Flexible pooling options
- Configurable activation
- Efficient tensor operations

## Implementation Notes
- Uses PyTorch's nn.Module
- Efficient parallel processing
- Memory-efficient concatenation
- Flexible configuration
- Clean interface

## Advantages
- Captures multiple temporal scales
- Efficient parallel processing
- Flexible architecture
- Easy to integrate
- Memory efficient

## Use Cases
- Signal processing
- Feature extraction
- Pattern recognition
- Multi-scale analysis
- Temporal modeling 