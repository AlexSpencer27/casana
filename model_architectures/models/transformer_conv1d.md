# Transformer Conv1D Model

> TL;DR: The attention-seeking model - it's always looking at the big picture, using transformer architecture to understand long-range relationships in signals while maintaining computational efficiency.

## Overview
The Transformer Conv1D model combines convolutional feature extraction with transformer-based sequence processing. It uses positional encoding and self-attention mechanisms to capture long-range dependencies in the signal while maintaining efficient processing through initial dimensionality reduction.

## Architecture

### Key Components
1. **Initial Feature Extraction**
   - First convolution (7x1)
   - 32 output channels
   - Stride: 2
   - Max pooling (size 2)
   - Second convolution (5x1)
   - 64 output channels
   - Stride: 2

2. **Positional Encoding**
   - Sinusoidal encoding
   - Dropout rate: 0.1
   - Maximum length: 5000
   - Learned position embeddings

3. **Transformer Encoder**
   - 2 encoder layers
   - 4 attention heads
   - Embedding dimension: 64
   - Feed-forward dimension: 128
   - Dropout rate: 0.1
   - Batch-first processing

4. **Global Attention Pooling**
   - Learned attention weights
   - Softmax normalization
   - Weighted sum pooling

5. **Prediction Head**
   - Two-layer MLP
   - Hidden dimensions: 128, 64
   - Dropout rate: 0.3
   - Peak ordering layer

### Data Flow
1. Input signal → Convolution blocks
2. Reshape for transformer
3. Add positional encoding
4. Apply transformer encoder
5. Global attention pooling
6. MLP prediction head
7. Peak ordering

## Technical Details

### Model Parameters
- Input: 1D signal
- Output: 3 values (peak1, midpoint, peak2)
- Channel dimensions:
  - First conv: 32
  - Second conv: 64
- Sequence length reduction: 8x (2x2x2)
- Hidden dimensions: 128, 64
- Dropout rates: 0.1, 0.3

### Key Features
- Transformer-based processing
- Positional encoding
- Global attention pooling
- Multi-head self-attention
- Peak ordering constraint

## Implementation Notes
- Uses PyTorch's nn.Module
- Inherits from BaseModel
- Implements gradient refinement capability
- Peak ordering ensures physical constraints
- Efficient sequence processing

## Advantages
- Captures long-range dependencies
- Position-aware processing
- Global context through attention
- Efficient feature extraction
- Physically meaningful output

## Use Cases
- Peak detection in time series data
- Signal processing applications
- Feature extraction from temporal data
- Pattern recognition in sequential data
- Long-range dependency modeling 