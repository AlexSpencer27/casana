import matplotlib.pyplot as plt
import torch

from src.config.config import config


def plot_predictions(signals: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor) -> None:
    """Plot the signals with their target and predicted peak positions.

    Args:
        signals: Batch of signals to plot
        targets: True peak positions
        predictions: Predicted peak positions
    """
    for i in range(config.visualization.num_predictions):
        plt.figure(figsize=(10, 4))
        plt.plot(signals[i].squeeze().numpy(), label="Original Signal")
        
        target_positions = targets[i].numpy() * config.signal.length
        predicted_pos = predictions[i].numpy() * config.signal.length

        plt.axvline(x=target_positions[0], color="g", linestyle="-", label="Peak 1")
        plt.axvline(x=target_positions[1], color="g", linestyle="--", label="Midpoint")
        plt.axvline(x=target_positions[2], color="g", linestyle="-", label="Peak 2")

        plt.axvline(x=predicted_pos[0], color="r", linestyle="-", label="Predicted Peak 1")
        plt.axvline(x=predicted_pos[1], color="r", linestyle="--", label="Predicted Midpoint")
        plt.axvline(x=predicted_pos[2], color="r", linestyle="-", label="Predicted Peak 2")

        plt.title(f"Example {i + 1}")
        plt.legend()
        plt.show() 