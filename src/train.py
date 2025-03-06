from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config.config import config
from src.utils.signal_generator import generate_batch
from src.models import get_model
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.training_monitor import TrainingMonitor


def main() -> None:
    # Create project directories
    best_results_dir = PROJECT_ROOT / 'best_model_results'
    best_results_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(PROJECT_ROOT)
    
    # Initialize training monitor
    monitor = TrainingMonitor(PROJECT_ROOT)
    
    # Get model class from registry and instantiate
    model_class = get_model(config.model.name)
    model = model_class()
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # training loop
    pbar = tqdm(range(config.training.num_epochs), desc="Training")
    for epoch in range(config.training.num_epochs):
        signals, targets = generate_batch()
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Update monitor
        monitor.update(loss.item(), epoch)
        
        # Update progress bar
        pbar.set_postfix(loss=loss.item())
        pbar.update(1)
    pbar.close()
            
    # Save final loss plot
    monitor.save_final_plot()

    final_loss = loss.item()

    # Evaluate model
    tracker.evaluate_model(model, final_loss)
    
    print("\nTraining complete! Check the 'experiments' directory for detailed metrics.")
    print("Best model results are maintained in the 'best_model_results' directory.")


if __name__ == "__main__":
    main() 