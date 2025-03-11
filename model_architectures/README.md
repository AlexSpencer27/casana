# Model Architectures Documentation

This directory contains comprehensive documentation for all models and components in the Casana project. We've streamlined our architectures to focus on what really matters - getting those peaks just right.

## Directory Structure
```
model_architectures/
├── models/              # Documentation for complete model architectures
│   ├── attention_dilated_conv1d.md
│   ├── pinn_peak_detector.md
│   ├── simplest_possible.md
│   └── base_model.md
└── components/         # Documentation for reusable components
    ├── spectral_branch.md
    ├── gradient_refinement.md
    ├── mlp_blocks.md
    ├── multi_scale_cnn.md
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

#### attention_dilated_conv1d.md
TL;DR: Multi-scale attention-based peak detection with signal processing priors
- Middle-tier model incorporating signal processing knowledge
- Multi-scale convolutions with attention mechanisms
- No heavy physics priors required

#### pinn_peak_detector.md
TL;DR: Physics-informed network with noise filtering and FFT-based template matching
- Noise filtering with high-pass filter
- 32 Hanning templates for peak detection
- Residual architecture with gradient refinement
- Guaranteed peak ordering

#### simplest_possible.md
TL;DR: Minimal fully-connected architecture for baseline peak detection performance
- Minimal assumptions about signal structure
- Two-layer fully connected architecture
- Light dropout regularization
- Perfect for baseline comparisons

#### base_model.md
TL;DR: Abstract base class defining common model interface and functionality
- Base class for all models
- Common functionality and interfaces
- Essential building blocks

### Components

#### spectral_branch.md
TL;DR: Frequency-domain analysis for enhanced peak discrimination
- Spectral analysis module
- Frequency domain processing
- Feature extraction from signal spectra

#### gradient_refinement.md
TL;DR: Gradient-based position refinement using local signal characteristics
- Gradient-based peak position refinement
- Fine-tuning of peak locations
- Physics-informed corrections

#### mlp_blocks.md
TL;DR: Configurable MLP blocks with skip connections and regularization
- Multi-layer perceptron building blocks
- Skip connections and dropout options
- Flexible layer configurations

#### multi_scale_cnn.md
TL;DR: Multi-scale convolutional processing for feature extraction at different resolutions
- Multi-scale convolutional components
- Various kernel sizes and dilations
- Feature extraction at different scales

#### common.md
TL;DR: Core utilities and shared functionality for model implementations
- Common utilities and shared components
- Helper functions and base classes
- Shared preprocessing tools

## Contributing

When adding new documentation:
1. Follow the established format
2. Include clear technical details
3. Provide implementation notes
4. Document advantages and use cases
5. Add visualizations where helpful