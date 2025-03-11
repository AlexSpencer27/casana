"""
Model registry for peak detection models.
All models are registered here and can be accessed by name.
"""

from typing import Dict, Type, Callable
from src.models.base_model import BaseModel

# Registry to store model classes
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(name: str) -> Callable:
    """
    Decorator to register a model class in the registry.
    
    Args:
        name: Name of the model to register
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str) -> Type[BaseModel]:
    """
    Get a model class by name.
    
    Args:
        name: Name of the model to get
        
    Returns:
        Model class
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]

# Import our three models representing different complexity levels
from src.models.simplest_possible import SimplestPossible  # Baseline model
from src.models.attention_dilated_conv1d import AttentionDilatedConv1D  # Middle-tier model
from src.models.pinn_peak_detector import PINNPeakDetector  # Physics-informed model

__all__ = [
    'register_model',
    'get_model',
    'SimplestPossible',
    'AttentionDilatedConv1D',
    'PINNPeakDetector'
]