import torch
import torch.nn as nn

from src.config.config import config


class PeakDetectionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement the model architecture
        self.fc_layer = nn.Linear(config.signal.length, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        x = self.fc_layer(x).view(config.training.batch_size, -1)
        return x 