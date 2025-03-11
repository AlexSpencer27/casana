# MLP Blocks

TL;DR: "I'm just a bunch of neurons doing honest work! 🧱"

## Overview
A collection of Multi-Layer Perceptron (MLP) building blocks with various architectures, including skip connections and progressive layouts. Designed for flexibility and performance.

## Architecture

### Core Components
1. Skip Connection MLP
   - Main path with two layers
   - Skip connection path
   - Optional batch normalization
   - Dropout regularization

2. Progressive MLP
   - Variable depth architecture
   - Configurable layer dimensions
   - Optional batch normalization
   - Progressive dropout

### Common Features
- ReLU activations
- Batch normalization (optional)
- Dropout regularization
- Residual connections

## Technical Details

### Input/Output Specifications
Skip Connection MLP:
- Input: `[batch_size, input_dim]`
- Hidden: `[batch_size, hidden_dim]`
- Output: `[batch_size, output_dim]`

Progressive MLP:
- Input: `[batch_size, layer_dims[0]]`
- Output: `[batch_size, layer_dims[-1]]`

### Key Parameters
- Default dropout rate: 0.3
- Batch normalization: True by default
- Configurable dimensions
- Flexible layer depths

## Implementation Notes

### Dependencies
- PyTorch
- torch.nn.functional

### Integration Guidelines
1. Choose appropriate architecture:
   - Skip Connection: For residual learning
   - Progressive: For deep architectures
2. Configure batch normalization based on needs
3. Adjust dropout rate for regularization
4. Set dimensions based on feature sizes

## Advantages
- Flexible architectures
- Strong regularization
- Residual learning support
- Configurable depth
- Easy integration

## Use Cases
- Feature transformation
- Deep representation learning
- Classification heads
- Embedding generation
- Signal processing 