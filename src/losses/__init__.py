from typing import Dict, Type
import torch.nn as nn

# Registry to store loss function classes
LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_loss(name: str):
    """Decorator to register a loss function class"""
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def get_loss(name: str) -> Type[nn.Module]:
    """Get a loss function class by name"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss function {name} not found in registry. Available losses: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]

# Import loss functions to register them
from .simple_mse import SimpleMSELoss
from .gradient_aware_loss import GradientAwareLoss 