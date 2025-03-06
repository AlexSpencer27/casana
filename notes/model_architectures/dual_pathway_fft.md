# DualPathwayFFT: A Dual-Domain Approach to Peak Detection

## Architecture Overview

DualPathwayFFT leverages both time and frequency domain representations to accurately detect peak positions in complex signals. The architecture employs parallel processing pathways that capture complementary signal characteristics.

## Key Components

### Multi-scale Time Domain Path

- Three parallel convolutional branches with varying kernel sizes (7, 15, 31)
- Captures peaks of different widths without prior knowledge of peak shape
- Progressive feature refinement via subsequent convolution layers

### Windowed FFT Frequency Domain Path

- Overlapping Hann-windowed FFT segments (256-point, 128 stride)
- Separate processing for low and high-frequency bands
- Specialized handling of real and imaginary components

### Cross-Domain Attention Mechanism

- Dynamically weights time vs. frequency features based on signal characteristics
- Learns optimal domain representation for different noise conditions

### Position-Aware Output Layer

- Soft constraints ensure logical peak ordering (peak1 < midpoint < peak2)
- Maintains gradient flow through differentiable sorting approach

## Design Rationale

The architecture addresses the specific characteristics of the input signal:

- Time domain processing captures spatial peak patterns
- Frequency domain processing separates sinusoidal noise
- Multi-scale processing accommodates random peak widths (10-40 samples)
- Reduced pooling preserves positional accuracy

## Limitations

- Computational Complexity - Dual pathways and FFT operations increase processing time
- Training Data Dependency - Performance may degrade on signals with noise characteristics different from training data
- Fixed Architecture - Pre-defined filter sizes may not adapt to extreme peak width variations
- Position Precision Tradeoff - Even with reduced pooling, some position resolution is sacrificed for feature abstraction
- Memory Requirements - The multiple parallel pathways increase the model's memory footprint

This hybrid approach demonstrates how domain knowledge can inform neural network design, creating a robust detector that performs well even with complex noise patterns.