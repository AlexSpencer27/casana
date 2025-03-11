# Multi-Scale CNN

TL;DR: "I look at your signal from every possible angle! 🔍"

## Overview
A convolutional neural network component that processes signals at multiple scales simultaneously using parallel convolutions with different kernel sizes. Can operate as a feature extractor or complete classifier.

## Architecture

### Core Components
1. Multi-Scale Convolutions
   - Parallel convolution layers
   - Different kernel sizes (7, 15, 31)
   - Shared channels per kernel (16)
   - Padding for length preservation

2. Feature Processing
   - Channel concatenation
   - Second conv (64 channels, kernel=9)
   - Third conv (64 channels, kernel=5)
   - Max pooling (stride=2)
   - Adaptive average pooling (16)

3. Classification Head (Optional)
   - Input: 64 * 16 features
   - Hidden layer: 256 units
   - Skip connection: 64 units
   - Batch normalization
   - Dropout (0.3)
   - Output: 3 units with sigmoid

## Technical Details

### Input/Output Specifications
Feature Extractor Mode:
- Input: `[batch_size, in_channels, signal_length]`
- Output: `[batch_size, channels_per_kernel * num_kernels, signal_length]`

Classification Mode:
- Input: `[batch_size, in_channels, signal_length]`
- Output: `[batch_size, 3]` (sigmoid activated)

### Key Parameters
- Input channels: 1 (default)
- Channels per kernel: 16
- Kernel sizes: (7, 15, 31)
- Second conv: 64 channels, kernel=9
- Third conv: 64 channels, kernel=5
- Pooling options: max, avg, none
- Dropout rate: 0.3
- Feature dimensions: 64, 256

## Implementation Notes

### Dependencies
- PyTorch
- torch.nn.functional

### Integration Guidelines
1. Choose operating mode:
   - Feature extractor: Set feature_extractor_mode=True
   - Classifier: Set feature_extractor_mode=False
2. Configure kernel sizes for application
3. Select appropriate pooling strategy:
   - 'max': For peak feature detection
   - 'avg': For smooth feature maps
   - None: For full resolution
4. Adjust channels based on complexity needs

## Advantages
- Multi-scale feature extraction
- Flexible operating modes
- Skip connection support
- Configurable architecture
- Efficient implementation

## Use Cases
- Signal feature extraction
- Multi-scale analysis
- Pattern recognition
- Time series processing
- Signal classification 