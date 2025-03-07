import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """Base class for all loss functions"""
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Loss value
        """
        pass 