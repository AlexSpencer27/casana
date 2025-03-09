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

# Import all models to register them
from src.models.simplest_possible import SimplestPossible
from src.models.multi_scale_conv1d import MultiScaleConv1D
from src.models.attention_dilated_conv1d import AttentionDilatedConv1D
from src.models.transformer_conv1d import TransformerConv1D
from src.models.spectral_peak_detector import SpectralPeakDetector
from src.models.dual_pathway_fft import DualPathwayFFT
from src.models.pinn_peak_detector import PINNPeakDetector

# All models are now refactored to use the component architecture
__all__ = [
    'register_model',
    'get_model',
    'SimplestPossible',
    'MultiScaleConv1D',
    'AttentionDilatedConv1D',
    'TransformerConv1D',
    'SpectralPeakDetector',
    'DualPathwayFFT',
    'PINNPeakDetector',
]