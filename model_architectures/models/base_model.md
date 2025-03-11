# Base Model

TL;DR: Abstract base class defining common model interface and functionality

## Overview
The BaseModel serves as the foundation for all models in the Casana project. It provides a consistent interface and common functionality that all models must implement.

## Architecture
- Inherits from PyTorch's `nn.Module` and Python's `ABC` (Abstract Base Class)
- Defines the basic structure that all models must follow
- Enforces implementation of key methods through abstract methods

## Technical Details

### Input/Output Specifications
- Input: PyTorch tensor of shape `[batch_size, signal_length]` or `[batch_size, channels, signal_length]`
- Output: PyTorch tensor containing model predictions

### Key Methods
- `forward(x: torch.Tensor) -> torch.Tensor`: Abstract method that must be implemented by all models
- `get_num_parameters() -> int`: Utility method to count trainable parameters

## Implementation Notes

### Dependencies
- PyTorch (`torch.nn`)
- Python's `abc` module for abstract base class functionality

### Integration Guidelines
1. All new models must inherit from `BaseModel`
2. Must implement the `forward()` method
3. Can access parameter counting through `get_num_parameters()`

## Advantages
- Ensures consistent interface across all models
- Provides common utility functions
- Makes model interchangeability possible
- Simplifies testing and integration 