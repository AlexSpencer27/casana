# Model Architectures Documentation

This directory contains comprehensive documentation for all models and components in the Casana project. We've streamlined our architectures to focus on what really matters - getting those peaks just right! 🎯

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
TL;DR: "I pay attention to ALL the peaks, but I'm not obsessed with physics. I just do what works! 🧠"
- Middle-tier model incorporating signal processing knowledge
- Multi-scale convolutions with attention mechanisms
- No heavy physics priors required

#### pinn_peak_detector.md
TL;DR: "I'm the physics nerd who actually reads the textbooks. And yes, I'll tell you why your peaks are wrong! 🤓"
- Physics-informed neural network
- Hanning template matching
- Spectral analysis and gradient-based refinement
- Early/late region specialization

#### simplest_possible.md
TL;DR: "Keep It Simple, Silly! Sometimes less is more... or at least it's something! 🤷‍♂️"
- Minimal assumptions about signal structure
- Two-layer fully connected architecture
- Light dropout regularization
- Perfect for baseline comparisons

#### base_model.md
TL;DR: "I'm the template everyone copies from. Not much to see here! 🏗️"
- Base class for all models
- Common functionality and interfaces
- Essential building blocks

### Components

#### spectral_branch.md
TL;DR: "I see frequencies you don't even know exist! 🌈"
- Spectral analysis module
- Frequency domain processing
- Feature extraction from signal spectra

#### gradient_refinement.md
TL;DR: "Trust me, that peak is *slightly* to the left... 🎯"
- Gradient-based peak position refinement
- Fine-tuning of peak locations
- Physics-informed corrections

#### mlp_blocks.md
TL;DR: "I'm just a bunch of neurons doing honest work! 🧱"
- Multi-layer perceptron building blocks
- Skip connections and dropout options
- Flexible layer configurations

#### multi_scale_cnn.md
TL;DR: "I look at your signal from every possible angle! 🔍"
- Multi-scale convolutional components
- Various kernel sizes and dilations
- Feature extraction at different scales

#### common.md
TL;DR: "I'm the utility belt - everyone needs me! 🛠️"
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