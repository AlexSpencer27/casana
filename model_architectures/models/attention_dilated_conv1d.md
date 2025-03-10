# Attention Dilated Conv1D Model

> TL;DR: The long-sighted attention seeker - it's got eyes everywhere (literally, with dilated convolutions) and knows exactly what to focus on, thanks to its channel-wise attention mechanism.

## Overview
The Attention Dilated Conv1D model is a sophisticated neural network architecture designed for peak detection in time series data. It combines multi-scale processing, attention mechanisms, and dilated convolutions to effectively identify and localize peaks in signals.

## Architecture

### Key Components
1. **Multi-Scale CNN Branch**
   - Processes input at multiple temporal scales
   - Uses kernel sizes of 7, 15, and 31
   - 16 channels per kernel size
   - No initial pooling to preserve temporal information

2. **Channel Attention Mechanism**
   - Global average pooling
   - Two-layer MLP for channel-wise attention
   - Sigmoid activation for attention weights
   - Learns to focus on relevant feature channels

3. **Dilated Convolution Block**
   - Kernel size: 5
   - Dilation rate: 2
   - Padding: 4
   - 48 input channels → 64 output channels
   - ReLU activation

4. **Feature Processing Pipeline**
   - MaxPool1d layers (stride=2)
   - Standard convolution (kernel=5, padding=2)
   - Adaptive feature pooling to fixed size
   - Skip connection MLP block
   - Peak ordering layer

### Data Flow
1. Input signal → Multi-scale processing
2. Channel attention weighting
3. Pooling and dilated convolution
4. Feature extraction and pooling
5. MLP processing with skip connections
6. Peak ordering and output

## Technical Details

### Model Parameters
- Input: 1D signal
- Output: 3 values (peak1, midpoint, peak2)
- Hidden dimensions: 64, 256
- Dropout rate: 0.3

### Key Features
- Multi-scale temporal processing
- Channel-wise attention mechanism
- Dilated convolutions for increased receptive field
- Skip connections for gradient flow
- Peak ordering constraint

## Implementation Notes
- Uses PyTorch's nn.Module
- Inherits from BaseModel
- Implements gradient refinement capability
- Peak ordering ensures peak1 < midpoint < peak2

## Advantages
- Effective at capturing multi-scale temporal patterns
- Attention mechanism helps focus on relevant features
- Dilated convolutions provide large receptive field with fewer parameters
- Skip connections help with training stability
- Peak ordering ensures physically meaningful output

## Use Cases
- Peak detection in time series data
- Signal processing applications
- Feature extraction from temporal data
- Pattern recognition in sequential data 