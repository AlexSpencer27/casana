# Neural Network Components

This directory contains reusable neural network components that can be combined to create models.

## Overview

The component architecture allows for:
- Reusing common patterns across models
- Easier experimentation with different architectures
- Improved maintainability and readability
- Simplified creation of new model variants

## Available Components

### Multi-Scale CNN Branch

**File:** `multi_scale_cnn.py`

**Purpose:** Process input with parallel convolutions using different kernel sizes

**Usage Example:**
```python
from src.models.components import MultiScaleCNNBranch

time_branch = MultiScaleCNNBranch(
    in_channels=1, 
    kernel_sizes=(7, 15, 31), 
    channels_per_kernel=16
)
```

### Spectral Branch

**File:** `spectral_branch.py`

**Purpose:** Process signals in frequency domain using FFT

**Usage Example:**
```python
from src.models.components import SpectralBranch

freq_branch = SpectralBranch(
    signal_length=2048,
    out_features=64,
    window_size=256
)
```

### Gradient Refinement Module

**File:** `gradient_refinement.py`

**Purpose:** Refine peak positions using gradient information

**Usage Example:**
```python
from src.models.components import GradientRefinementModule

gradient_refiner = GradientRefinementModule(
    signal_length=2048,
    num_iterations=3
)
```

### Peak Ordering Layer

**File:** `common.py`

**Purpose:** Ensure consistent ordering of peak outputs (peak1 < midpoint < peak2)

**Usage Example:**
```python
from src.models.components import PeakOrderingLayer

peak_ordering = PeakOrderingLayer()
```

### Skip Connection MLP

**File:** `mlp_blocks.py`

**Purpose:** Fully connected layers with residual connections

**Usage Example:**
```python
from src.models.components import SkipConnectionMLP

mlp_block = SkipConnectionMLP(
    input_dim=128, 
    hidden_dim=256, 
    output_dim=64
)
```

### Attention Module

**File:** `attention.py`

**Purpose:** Apply attention mechanisms to features

**Usage Example:**
```python
from src.models.components import AttentionModule

attention = AttentionModule(
    feature_dim=64,
    attention_type='channel'
)
```

## Creating New Models

To create a new model using these components:

1. Import the necessary components
2. Combine them in your model's `__init__` method
3. Connect them in the `forward` method

See `dual_pathway_fft_v2.py` for an example of a model built with components.

## Extending the Component Library

To add a new component:

1. Create a new file or add to an existing one based on component type
2. Implement the component as a subclass of `nn.Module`
3. Add it to the exports in `__init__.py`
4. Document it in this README 