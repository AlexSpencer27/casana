# MLP Blocks Component

## Overview
The MLP Blocks component provides flexible and efficient implementations of Multi-Layer Perceptron architectures. It includes two main variants: SkipConnectionMLP with residual connections and ProgressiveMLP with configurable layer dimensions.

## Architecture

### Key Components
1. **SkipConnectionMLP**
   - Residual connections
   - Two-layer main path
   - Parallel skip connection
   - Batch normalization option
   - Dropout regularization

2. **ProgressiveMLP**
   - Configurable layer dimensions
   - Progressive layer structure
   - Optional batch normalization
   - Dropout regularization
   - ReLU activation

### Data Flow
1. **SkipConnectionMLP**
   - Input → Main path (FC1 → BN → ReLU → Dropout → FC2 → BN → ReLU)
   - Input → Skip path (FC_skip → ReLU)
   - Combine paths → Dropout

2. **ProgressiveMLP**
   - Input → Layer1 → BN → ReLU → Dropout
   - → Layer2 → BN → ReLU → Dropout
   - → ... → LayerN (no BN/ReLU)

## Technical Details

### Module Parameters
#### SkipConnectionMLP
- Input dimension: configurable
- Hidden dimension: configurable
- Output dimension: configurable
- Dropout rate: 0.3 (default)
- Batch normalization: optional

#### ProgressiveMLP
- Layer dimensions: configurable list
- Dropout rate: 0.3 (default)
- Batch normalization: optional

### Key Features
- Residual connections
- Batch normalization
- Dropout regularization
- Flexible architecture
- Efficient computation

## Implementation Notes
- Uses PyTorch's nn.Module
- Efficient tensor operations
- Clean interface
- Memory efficient
- Flexible configuration

## Advantages
- Improved gradient flow
- Better training stability
- Flexible architecture
- Easy to integrate
- Memory efficient

## Use Cases
- Feature processing
- Dimensionality reduction
- Pattern recognition
- Feature fusion
- Classification tasks 