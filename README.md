# Neural Network Signal Peak Detection

## Overview
A neural network framework for detecting peaks in noisy signals, focusing on modularity and reproducibility.

## Key Principles
### 1. Experiment Pipeline
- Central configuration via YAML
- Structured output directory
- Traceable models and training (simplified for this exercise)

### 2. Component Architecture
- Reusable building blocks
- Easy to extend
- Simple baseline to advanced implementations

### 3. Experiment Monitoring
- Loss & component visualizations
- Early stopping with configurable patience
- Curriculum parameter tracking:
  - Learning rate progression
  - Noise amplitude changes
  - Loss weight adjustments

### 4. Result Tracking
- Metrics collection
- Standardized visualizations
- Cross-experiment comparison

## Problem Analysis

### Initial Challenge Decomposition
- Signal characteristics vary significantly across samples:
  - Peak widths: 10-40 samples 
  - Amplitude ratios: 0.8->1.0 between peaks
  - Inter-peak distance: Variable from 0.1-1.8 time units
- Noise components:
  - Complex multi-frequency interference
  - Amplitude-varying distortions
  - Non-uniform baseline drift

## Solution Evolution

### 1. Basic Feed-Forward Network
Initial Approach:
- Simple fully-connected architecture
- Direct signal-to-position mapping
- Minimal assumptions about signal structure

Limitations:
- No spatial hierarchy understanding
- Sensitive to peak amplitude variations

### 2. CNN with Attention
Motivation:
- Need for translation invariance
- Multi-scale feature detection
- Focus on relevant signal regions

Design Evolution:
- Added convolutional layers for spatial features
- Incorporated attention mechanism

Challenges:
- Training instability with attention
- Computational overhead
- Complex hyperparameter tuning

### 3. Physics-Informed Approach
Integration Strategy:
- Spectral domain constraints
- Gradient-based peak properties
- Signal ~physics priors

Design Considerations:
- Balance between data-driven and physics-based components
- Integration of domain knowledge
- Computational efficiency trade-offs

Results:
- Robust to unseen noise patterns
- Faster convergence during training

## Design Decisions

### Modular Architecture
Rationale:
- Component isolation for testing
- Easy experimentation with architectures
- Reusable building blocks across models

### Curriculum Learning Design
Progression Strategy:
- Start: Clean signals, well-separated peaks
- Middle: Introduce gaussian noise, varying SNR
- End: Complex noise, overlapping peaks

Implementation:
```python
def noise_schedule(epoch):
    return min(MAX_NOISE, BASE_NOISE * (1 + epoch/NOISE_RAMP_EPOCHS))
```

### Signal Generation
Design Choices:
- Parameterized peak shapes for variety
- Physics-based noise models
- Controlled difficulty progression

Validation:
- Statistical analysis of generated signals
- Coverage of edge cases
- Real-world signal comparison

## Future Considerations

### Scaling Strategy
1. Performance Optimization
   - ONNX runtime deployment
   - Quantization-aware training
   - Batch size optimization

2. Architecture Extensions
   - Multi-resolution processing
   - Uncertainty estimation
   - Online learning capabilities

3. Known Limitations
   - Peak overlap handling
   - Extreme noise scenarios
   - Computational complexity scaling

## Problem Statement
Given a signal with two peaks and noise, predict:
- First peak position
- Midpoint between peaks
- Second peak position

## Architecture

### Models
Three implementations with increasing complexity ([detailed documentation](model_architectures/)):

1. **Basic Feed-Forward Network** ([docs](model_architectures/models/simplest_possible.md))
   > Pure data-driven approach with minimal architecture
   > - Simple fully-connected layers
   > - No signal processing assumptions
   > - Baseline performance benchmark

2. **CNN with Attention** ([docs](model_architectures/models/attention_dilated_conv1d.md))
   > Advanced architecture, still pure data-driven
   > - Multi-scale CNN + attention mechanisms
   > - Learns signal patterns from data
   > - No explicit signal priors

3. **Physics-Informed Network** ([docs](model_architectures/models/pinn_peak_detector.md))
   > Hybrid approach combining deep learning with signal processing
   > - Spectral analysis integration
   > - Gradient-based constraints
   > - Explicit signal priors

### Components
Building blocks ([component docs](model_architectures/components/)):

- **CNN Blocks** ([docs](model_architectures/components/multi_scale_cnn.md))
  > Multi-scale feature extraction
- **Spectral Processing** ([docs](model_architectures/components/spectral_branch.md))
  > Frequency-domain analysis
- **Peak Refinement** ([docs](model_architectures/components/gradient_refinement.md))
  > Gradient-based position refinement
- **Common Utilities** ([docs](model_architectures/components/common.md))
  > Shared functionality

### Loss Function
Combines ([source](src/losses/peak_loss.py)):
- Position accuracy
- Peak characteristics
- Curriculum learning

### Curriculum Learning
Configurable curriculum options ([config.yaml](config.yaml)):
- Complex noise: Control noise amplitude progression
- Learning rate: Scheduled learning rate decay
- Loss weights: Adjustable component weight progression
  - Position/magnitude weights
  - Gradient constraints
  - Second derivative terms

Current implementation uses static weights, but infrastructure supports dynamic curriculum learning.

## Project Structure
```
.
├── config.yaml           # Configuration
├── pyproject.toml       # Dependencies
├── multi_experiment_run/  # Multi-model experiment framework
│   ├── multi_config.yaml # Experiment configurations
│   └── run_experiments.py # Experiment runner
├── best_model_results/   # Maintained experiment results
│   ├── model_comparison.png  # Performance visualization
│   └── {model_name}/    # Per-model results
└── src/
    ├── config/         
    ├── models/          
    │   └── components/  
    ├── utils/           
    └── train.py         
```

Results from experiments are maintained in `best_model_results/`, providing performance comparisons and detailed per-model metrics.

## Usage

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
pip install torch
```

### Running Experiments
```bash
# Single experiment
python src/train.py

# Multiple experiments
python multi_experiment_run/run_experiments.py
```

### Configuration
```yaml
# config.yaml - Single experiment
training:
  batch_size: 32
  num_epochs: 2000
signal:
  length: 2048
  sampling_rate: 1024

# multi_config.yaml - Multiple experiments
all experiments will inherit base_config parameters,
then you can inject experimental ones for each
base_config:
  training:
    batch_size: 256
experiments:
  - name: "baseline"
    model:
      name: "simplest_possible"
  - name: "cnn_attention"
    model:
      name: "attention_dilated_conv1d"
```

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- PyYAML
