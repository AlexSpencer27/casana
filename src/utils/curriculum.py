from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.optim import Optimizer

from src.config.config import config


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning aspects."""
    # Complex noise progression
    noise_start: float = 0.0
    noise_end: float = 0.2
    noise_min: float = 0.05
    noise_epochs: int = 10000
    
    # Learning rate scheduling
    lr_start: float = 5e-5
    lr_end: float = 1e-6
    lr_epochs: int = 50000  # Epochs to reach minimum learning rate
    
    # Loss weights progression
    loss_weights_start: Dict[str, float] = None
    loss_weights_end: Dict[str, float] = None
    loss_weights_epochs: int = 20000
    
    def __post_init__(self):
        # Set default loss weight progressions if not provided
        if self.loss_weights_start is None:
            self.loss_weights_start = {
                'position': 100.0,
                'magnitude': 1.0,
                'gradient': 1e-8,
                'second_derivative': 1e-4
            }
        if self.loss_weights_end is None:
            self.loss_weights_end = {
                'position': 100.0,
                'magnitude': 1.0,
                'gradient': 1e-6,
                'second_derivative': 1e-2
            }


class CurriculumManager:
    """Manages all aspects of curriculum learning during training."""
    
    def __init__(self, optimizer: Optimizer):
        """Initialize curriculum manager.
        
        Args:
            optimizer: The optimizer whose learning rate will be managed
        """
        self.optimizer = optimizer
        
        # Initialize from config
        self._init_from_config()
    
    def _init_from_config(self):
        """Initialize progression parameters from config."""
        # Complex noise settings
        self.noise_start = config.curriculum.complex_noise.start_amplitude
        self.noise_end = config.curriculum.complex_noise.end_amplitude
        self.noise_min = config.curriculum.complex_noise.min_amplitude
        self.noise_epochs = config.curriculum.complex_noise.epochs_to_max
        
        # Learning rate settings
        self.lr_start = config.curriculum.learning_rate.start
        self.lr_end = config.curriculum.learning_rate.end
        self.lr_epochs = config.curriculum.learning_rate.epochs_to_min
        
        # Loss weights
        self.loss_weights_start = {
            'position': config.curriculum.loss_weights.start.position,
            'magnitude': config.curriculum.loss_weights.start.magnitude,
            'gradient': config.curriculum.loss_weights.start.gradient,
            'second_derivative': config.curriculum.loss_weights.start.second_derivative
        }
        self.loss_weights_end = {
            'position': config.curriculum.loss_weights.final.position,
            'magnitude': config.curriculum.loss_weights.final.magnitude,
            'gradient': config.curriculum.loss_weights.final.gradient,
            'second_derivative': config.curriculum.loss_weights.final.second_derivative
        }
        self.loss_weights_epochs = config.curriculum.loss_weights.epochs_to_final
    
    def get_noise_amplitude(self, epoch: int) -> float:
        """Get the current noise amplitude based on training progress."""
        progress = min(epoch / self.noise_epochs, 1.0)
        return self.noise_start + (self.noise_end - self.noise_start) * progress
    
    def update_learning_rate(self, epoch: int):
        """Update the learning rate based on training progress."""
        progress = min(epoch / self.lr_epochs, 1.0)
        current_lr = self.lr_start * (self.lr_end / self.lr_start) ** progress
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
    
    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Get the current loss weights based on training progress."""
        progress = min(epoch / self.loss_weights_epochs, 1.0)
        
        current_weights = {}
        for key in self.loss_weights_start.keys():
            start_val = self.loss_weights_start[key]
            end_val = self.loss_weights_end[key]
            
            # Use exponential interpolation for very small values
            if min(start_val, end_val) < 1e-3:
                current_weights[key] = start_val * (end_val / start_val) ** progress
            else:
                # Linear interpolation for larger values
                current_weights[key] = start_val + (end_val - start_val) * progress
        
        return current_weights
    
    def step(self, epoch: int) -> Dict[str, float]:
        """Perform all curriculum updates for the current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dict containing current values for monitoring
        """
        self.update_learning_rate(epoch)
        noise_amplitude = self.get_noise_amplitude(epoch)
        loss_weights = self.get_loss_weights(epoch)
        
        # Update config with current loss weights
        config.loss.position_weight = loss_weights['position']
        config.loss.magnitude_weight = loss_weights['magnitude']
        config.loss.gradient_weight = loss_weights['gradient']
        config.loss.second_derivative_weight = loss_weights['second_derivative']
        
        return {
            'noise': 'off' if noise_amplitude < self.noise_min else f"{noise_amplitude:.3f}",
            'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            'grad_weight': f"{loss_weights['gradient']:.2e}",
            'second_deriv_weight': f"{loss_weights['second_derivative']:.2e}"
        } 