from pathlib import Path
import sys
import os

# Add project root to path
def find_project_root() -> Path:
    """Find the project root by looking for config.yaml"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'config.yaml').exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (config.yaml)")

# Add project root to Python path
project_root = find_project_root()
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim

from src.config.config import config
from src.utils.signal_generator import generate_batch
from src.models import get_model
from src.utils.plotter import plot_predictions
from src.utils.training_monitor import TrainingMonitor


def main() -> None:
    # Create figures directory if it doesn't exist
    figures_dir = project_root / 'figures'
    
    # Initialize training monitor
    monitor = TrainingMonitor(figures_dir)
    
    # Get model class from registry and instantiate
    model_class = get_model(config.model.name)
    model = model_class()
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # training loop
    for epoch in range(config.training.num_epochs):
        signals, targets = generate_batch()
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Update monitor
        monitor.update(loss.item(), epoch)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{config.training.num_epochs}], Loss: {monitor.current_loss:.6f}")

    # Save final loss plot
    monitor.save_final_plot()

    # predictions
    # signals, targets = generate_batch()
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(signals)

    # plot predictions
    # plot_predictions(signals, targets, predictions)


if __name__ == "__main__":
    main() 