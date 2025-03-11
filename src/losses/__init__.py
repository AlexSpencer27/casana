"""
Loss functions for training models.
"""

from typing import Callable, Dict, Type
from src.losses.base_loss import BaseLoss

# Registry to store loss functions
_LOSS_REGISTRY: Dict[str, Type[BaseLoss]] = {}

def register_loss(name: str) -> Callable:
    """Decorator to register a loss function."""
    def _register(cls: Type[BaseLoss]) -> Type[BaseLoss]:
        _LOSS_REGISTRY[name] = cls
        return cls
    return _register

def get_loss(name: str) -> Type[BaseLoss]:
    """Get a loss function by name."""
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
    return _LOSS_REGISTRY[name]

# Import all loss functions
from src.losses.gradient_aware_loss import PeakLoss

__all__ = ['get_loss', 'register_loss', 'PeakLoss']

# Import loss functions to register them
from .simple_mse import SimpleMSELoss
from .gradient_aware_loss import GradientAwareLoss 