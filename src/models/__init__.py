from typing import Dict, Type
import torch.nn as nn

# Registry to store model classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """Decorator to register a model class"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str) -> Type[nn.Module]:
    """Get a model class by name"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]

# Import models to register them
from .dual_pathway_fft import DualPathwayFFT 
from .simplest_possible import SimplestPossible
from .multi_scale_conv1d import MultiScaleConv1D
from .attention_dilated_conv1d import AttentionDilatedConv1D
from .transformer_conv1d import TransformerConv1D
from .spectral_peak_detector import SpectralPeakDetector