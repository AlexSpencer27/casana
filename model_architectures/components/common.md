# Common Utilities Component

TL;DR: Core utilities and shared functionality for model implementations

## Overview
A collection of common utilities and components used across different models. Currently focuses on adaptive feature pooling for handling variable-length inputs.

## Architecture

### Core Components
1. Adaptive Feature Pooling
   - Flexible input size handling
   - Configurable output size
   - Multiple pooling types
   - Channel-wise operation

### Features
- Automatic size adaptation
- Preserves channel dimensions
- Memory efficient
- Batch processing support

## Technical Details

### Input/Output Specifications
AdaptiveFeaturePooling:
- Input: `[batch_size, channels, length]`
- Output: `[batch_size, channels, output_size]`

### Key Parameters
- Output size: 16 (default)
- Pooling types: 
  - 'avg': Average pooling
  - 'max': Max pooling

## Implementation Notes

### Dependencies
- PyTorch
- torch.nn

### Integration Guidelines
1. Choose appropriate pooling type:
   - Average: For smooth feature maps
   - Max: For peak feature preservation
2. Set output size based on needs
3. Can be used in any part of network
4. Handles variable input lengths

## Advantages
- Flexible input handling
- Multiple pooling options
- Simple integration
- Efficient implementation
- Batch processing support

## Use Cases
- Feature map resizing
- Variable length handling
- Dimension reduction
- Memory optimization
- Model architecture flexibility 