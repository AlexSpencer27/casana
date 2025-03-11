"""
Neural network components for building models.
This module contains reusable building blocks that can be combined to create models.
"""

# Import components needed for our three models
from .multi_scale_cnn import MultiScaleCNNBranch  # Used by AttentionDilatedConv1D
from .spectral_branch import SpectralBranch  # Used by PINN
from .gradient_refinement import GradientRefinementModule  # Used by PINN
from .common import AdaptiveFeaturePooling  # Used by AttentionDilatedConv1D
from .mlp_blocks import SkipConnectionMLP  # Used by AttentionDilatedConv1D

# Export components
__all__ = [
    'MultiScaleCNNBranch',
    'SpectralBranch',
    'GradientRefinementModule',
    'AdaptiveFeaturePooling',
    'SkipConnectionMLP'
] 