# Multi-Scale Conv1D Model

> TL;DR: The efficient multitasker - it processes signals at multiple scales simultaneously, like having multiple pairs of eyes looking at different levels of detail all at once.

## Overview
The Multi-Scale Conv1D model is a streamlined architecture that processes signals at multiple temporal scales using parallel convolutional branches. It combines efficient feature extraction with skip connections to achieve robust peak detection while maintaining computational efficiency.

## Architecture

### Key Components
1. **Multi-Scale CNN Branch**
   - Parallel convolution processing
   - Kernel sizes: 7, 15, 31
   - 16 channels per kernel
   - No initial pooling
   - Total channels: 48

2. **Convolution Blocks**
   - Second convolution (9x1)
   - Padding: 4
   - Output channels: 64
   - Max pooling (size 2)

3. **Third Convolution Block**
   - Kernel size: 5
   - Padding: 2
   - Output channels: 64
   - Adaptive pooling to fixed size

4. **Feature Processing Pipeline**
   - Skip connection MLP block
   - Hidden dimension: 256
   - Output dimension: 64
   - Dropout rate: 0.3
   - Peak ordering layer

### Data Flow
1. Input signal → Multi-scale processing
2. Second convolution block
3. Third convolution block
4. Adaptive pooling
5. Skip connection MLP
6. Peak ordering and output

## Technical Details

### Model Parameters
- Input: 1D signal
- Output: 3 values (peak1, midpoint, peak2)
- Channel dimensions:
  - Multi-scale: 48
  - Second conv: 64
  - Third conv: 64
- Hidden dimensions: 256, 64
- Dropout rate: 0.3

### Key Features
- Multi-scale temporal processing
- Efficient feature extraction
- Skip connections
- Peak ordering constraint
- Adaptive pooling

## Implementation Notes
- Uses PyTorch's nn.Module
- Inherits from BaseModel
- Implements gradient refinement capability
- Peak ordering ensures peak1 < midpoint < peak2
- Efficient parallel processing

## Advantages
- Fast inference time
- Memory efficient
- Robust feature extraction
- Skip connections improve training
- Physically meaningful output

## Use Cases
- Peak detection in time series data
- Signal processing applications
- Feature extraction from temporal data
- Pattern recognition in sequential data
- Real-time processing 