# Simplest Possible Model

TL;DR: Minimal fully-connected architecture for baseline peak detection performance

## Overview
The most basic model that makes minimal assumptions about the signal structure. Uses a simple two-layer fully connected architecture with dropout regularization.

## Architecture

### Core Components
1. Input Processing
   - Flattening layer
   - No assumptions about signal structure

2. Neural Network
   - First layer: signal_length → 64
   - ReLU activation
   - Dropout (0.2)
   - Second layer: 64 → 3
   - Sigmoid output activation

## Technical Details

### Input/Output Specifications
- Input: Signal tensor of shape `[batch_size, signal_length]`
- Output: 3-dimensional prediction tensor (sigmoid activated)

### Key Parameters
- Hidden dimension: 64
- Dropout rate: 0.2
- Output dimension: 3

## Implementation Notes

### Dependencies
- PyTorch
- No custom components required

### Integration Guidelines
1. Signal can be any length (defined in config)
2. No preprocessing required beyond basic normalization
3. Output is always sigmoid-activated

## Advantages
- Extremely simple architecture
- Minimal assumptions
- Fast training and inference
- Perfect for baseline comparisons
- Easy to debug and understand

## Use Cases
- Baseline for peak detection
- Quick prototyping
- Performance comparison
- Educational purposes
- Simple signal processing 