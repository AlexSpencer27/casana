# Attention Dilated Conv1D

TL;DR: Advanced data-driven peak detection using multi-scale CNN and attention mechanisms

## Overview
A sophisticated model that leverages deep learning techniques through multi-scale convolutions and attention mechanisms, learning signal patterns purely from data without incorporating explicit signal processing priors.

## Architecture

### Core Components
1. Multi-scale CNN Branch
   - Multiple kernel sizes (7, 15, 31)
   - 16 channels per kernel
   - Feature extraction mode

2. Channel Attention
   - Adaptive average pooling
   - Two-layer MLP attention mechanism
   - Sigmoid activation for attention weights

3. Temporal Processing
   - Dilated convolution (kernel=5, dilation=2)
   - Max pooling for dimensionality reduction
   - Adaptive feature pooling

4. Classification Head
   - Skip-connection MLP
   - Dropout regularization (0.3)
   - Sigmoid output activation

## Technical Details

### Input/Output Specifications
- Input: Signal tensor of shape `[batch_size, signal_length]`
- Output: 3-dimensional prediction tensor with sigmoid activation

### Key Parameters
- Multi-scale kernel sizes: [7, 15, 31]
- Channels per kernel: 16
- Dilated conv: kernel=5, dilation=2
- Dropout rate: 0.3
- Output dimension: 3

## Implementation Notes

### Dependencies
- PyTorch
- Custom components:
  - MultiScaleCNNBranch
  - SkipConnectionMLP
  - AdaptiveFeaturePooling

### Integration Guidelines
1. Ensure signal is properly normalized
2. Input can be either 2D or 3D (handles channel dimension automatically)
3. Output is always sigmoid-activated for stable training

## Advantages
- Purely data-driven approach
- Effective multi-scale feature extraction
- Attention mechanism helps focus on relevant signal regions
- Learns patterns directly from data
- Good balance of complexity and performance

## Use Cases
- Peak detection in time series data
- Signal processing applications
- Feature extraction from temporal data
- Pattern recognition in sequential data 