import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.config.config import config
from src.models.components.gradient_refinement import GradientRefinementModule

class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    
    def __init__(self) -> None:
        super().__init__()
        # Initialize gradient refinement if enabled
        self.gradient_refinement_enabled = config.model.gradient_refinement.enabled
        if self.gradient_refinement_enabled:
            self.gradient_refiner = GradientRefinementModule(
                signal_length=config.signal.length,
                num_iterations=config.model.gradient_refinement.num_iterations,
                step_size=config.model.gradient_refinement.step_size
            )
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        pass
    
    def refine_peaks(self, signal: torch.Tensor, peaks: torch.Tensor) -> torch.Tensor:
        """Refine peak positions using gradient information if enabled"""
        if self.gradient_refinement_enabled:
            return self.gradient_refiner(signal, peaks)
        return peaks
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 