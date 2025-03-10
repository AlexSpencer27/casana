# Attention Components

## Overview
The attention components provide flexible attention mechanisms for feature weighting and selection in neural networks. The module includes two main classes: `AttentionModule` for general attention operations and `CrossPathwayAttention` for fusing time and frequency domain features.

## AttentionModule

### Architecture
The `AttentionModule` supports three types of attention mechanisms:

1. **Self-Attention**
   - Multi-head attention mechanism
   - Query, Key, Value projections
   - Scaled dot-product attention
   - Output projection

2. **Channel Attention**
   - Squeeze-and-excitation style
   - Two-layer MLP for channel-wise weighting
   - Sigmoid activation for attention weights

3. **Spatial Attention**
   - 1D convolutional attention
   - Two-layer Conv1d for spatial weighting
   - Sigmoid activation for attention weights

### Technical Details

#### Parameters
- `feature_dim`: Dimension of input features
- `attention_dim`: Dimension for attention calculation (default: 64)
- `num_heads`: Number of attention heads (default: 1)
- `attention_type`: Type of attention mechanism ('self', 'channel', 'spatial')

#### Input/Output Shapes
- Self-Attention: [batch_size, feature_dim] → [batch_size, feature_dim]
- Channel Attention: [batch_size, feature_dim] → [batch_size, feature_dim]
- Spatial Attention: [batch_size, channels, length] → [batch_size, channels, length]

## CrossPathwayAttention

### Architecture
- Fuses time and frequency domain features
- Learns attention weights for each domain
- Weighted combination of features

### Technical Details

#### Parameters
- `time_features`: Number of time domain features
- `freq_features`: Number of frequency domain features

#### Input/Output
- Input: Time and frequency features
- Output: Weighted combination of both feature sets

## Implementation Notes
- Built on PyTorch's nn.Module
- Supports batch processing
- Efficient tensor operations
- Flexible attention mechanisms

## Use Cases
- Feature weighting in neural networks
- Multi-head self-attention for sequence processing
- Channel-wise feature selection
- Spatial attention for 1D signals
- Time-frequency feature fusion

## Advantages
- Modular and reusable design
- Multiple attention mechanisms in one module
- Efficient implementation
- Flexible input/output shapes
- Easy integration into existing architectures 