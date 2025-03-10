# Model Architectures Documentation

This directory contains comprehensive documentation for all models and components in the Casana project. The documentation is organized into two main sections:

## Directory Structure
```
model_architectures/
├── models/              # Documentation for complete model architectures
│   ├── attention_dilated_conv1d.md
│   ├── dual_pathway_fft.md
│   ├── multi_scale_conv1d.md
│   ├── pinn_peak_detector.md
│   ├── spectral_peak_detector.md
│   ├── simplest_possible.md
│   └── transformer_conv1d.md
└── components/         # Documentation for reusable components
    ├── attention.md
    ├── gradient_refinement.md
    ├── mlp_blocks.md
    ├── multi_scale_cnn.md
    ├── spectral_branch.md
    └── common.md
```

## Documentation Format

Each documentation file follows a consistent structure:

1. **Overview**
   - High-level description of the model/component
   - Purpose and main use cases

2. **Architecture**
   - Detailed description of the architecture
   - Key components and their interactions
   - Data flow through the system

3. **Technical Details**
   - Model parameters and hyperparameters
   - Input/output specifications
   - Key features and capabilities

4. **Implementation Notes**
   - Implementation details
   - Dependencies and requirements
   - Integration guidelines

5. **Advantages**
   - Key benefits and strengths
   - Use cases and applications
   - Performance considerations

## Models

### Complete Architectures
- `attention_dilated_conv1d.md`: Attention-based dilated convolution model
- `dual_pathway_fft.md`: Dual pathway model with FFT processing
- `multi_scale_conv1d.md`: Multi-scale convolutional model
- `pinn_peak_detector.md`: Physics-informed neural network for peak detection
- `spectral_peak_detector.md`: Spectral domain peak detection model
- `simplest_possible.md`: Minimal implementation for baseline
- `transformer_conv1d.md`: Transformer-based convolutional model

### Components
- `attention.md`: Attention mechanisms and modules
- `gradient_refinement.md`: Gradient-based peak refinement
- `mlp_blocks.md`: Multi-layer perceptron building blocks
- `multi_scale_cnn.md`: Multi-scale convolutional components
- `spectral_branch.md`: Spectral processing components
- `common.md`: Common utilities and shared components

## Contributing

When adding new documentation:
1. Follow the established format
2. Include clear technical details
3. Provide implementation notes
4. Document advantages and use cases
5. Add visualizations where helpful 