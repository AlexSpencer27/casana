from pathlib import Path
import sys
import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config.config import config
from src.utils.signal_generator import generate_batch
from src.models import get_model
from src.losses import get_loss
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.training_monitor import TrainingMonitor


def main() -> None:
    # Enable anomaly detection right at the start
    torch.autograd.set_detect_anomaly(True)
    
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
    
    # Get loss function from registry and instantiate
    loss_class = get_loss(config.loss.name)
    criterion = loss_class()  
    
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # training loop
    pbar = tqdm(range(config.training.num_epochs), desc="Training")
    for epoch in range(config.training.num_epochs):
        signals, targets = generate_batch()
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, targets, signals)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        # Update monitor and check early stopping
        if not monitor.update(current_loss, epoch):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break
            
        # Update progress bar
        pbar.set_postfix(loss=current_loss, best_loss=monitor.best_loss_value, patience=monitor.patience_counter)
        pbar.update(1)
            
    pbar.close()
            
    # Save final loss plot
    monitor.save_final_plot()

    # Use best loss for evaluation
    tracker.evaluate_model(model, monitor.best_loss_value)
    
    print("\nTraining complete! Check the 'experiments' directory for detailed metrics.")
    print("Best model results are maintained in the 'best_model_results' directory.")


if __name__ == "__main__":
    main() 