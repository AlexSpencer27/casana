# Simplest Possible Model

> TL;DR: The minimalist's dream - just two layers and a dream, proving that sometimes less is more when it comes to peak detection.

## Overview
The Simplest Possible model serves as a baseline architecture for peak detection. It implements a minimal neural network design with just two fully connected layers, demonstrating the fundamental approach to the peak detection task while maintaining physical constraints through peak ordering.

## Architecture

### Key Components
1. **Input Processing**
   - Flatten layer for 1D signal
   - Input dimension: signal length
   - No spatial processing

2. **First Layer**
   - Fully connected layer
   - Input → 64 hidden units
   - ReLU activation
   - Dropout (0.2)

3. **Output Layer**
   - Fully connected layer
   - 64 → 3 output units
   - No activation (linear)

4. **Peak Ordering**
   - Ensures physical constraints
   - Softness parameter: 0.1
   - Maintains peak1 < midpoint < peak2

### Data Flow
1. Input signal → Flatten
2. First FC layer → ReLU → Dropout
3. Second FC layer
4. Peak ordering
5. Output

## Technical Details

### Model Parameters
- Input: 1D signal
- Output: 3 values (peak1, midpoint, peak2)
- Hidden dimension: 64
- Dropout rate: 0.2
- Peak ordering softness: 0.1

### Key Features
- Minimal architecture
- Fast inference
- Peak ordering constraint
- Light regularization
- No spatial processing

## Implementation Notes
- Uses PyTorch's nn.Module
- Inherits from BaseModel
- Implements gradient refinement capability
- Peak ordering ensures physical constraints
- Efficient forward pass

## Advantages
- Fastest inference time
- Minimal memory usage
- Easy to understand
- Good baseline performance
- Physically meaningful output

## Use Cases
- Baseline for peak detection
- Quick prototyping
- Performance comparison
- Educational purposes
- Simple signal processing 