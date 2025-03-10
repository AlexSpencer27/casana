from pathlib import Path
import sys
import torch
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config.config import config
from src.utils.signal_generator import generate_batch, generate_signal
from src.models import get_model
from src.losses import get_loss
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.training_monitor import TrainingMonitor
from src.utils.curriculum import CurriculumManager


def main() -> None:
    # Enable anomaly detection right at the start
    torch.autograd.set_detect_anomaly(True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    model.to(device)
    model.train()
    
    # Get loss function from registry and instantiate
    loss_class = get_loss(config.loss.name)
    criterion = loss_class().to(device)  
    
    # Initialize optimizer and curriculum manager
    optimizer = optim.Adam(model.parameters(), lr=config.curriculum.learning_rate.start)
    curriculum = CurriculumManager(optimizer)

    # Store current epoch for signal generator
    current_epoch = 0

    # Monkey patch the generate_signal function to use curriculum learning
    original_generate_signal = generate_signal
    def curriculum_generate_signal(*args, **kwargs):
        # Add current noise amplitude to signal generation
        kwargs['noise_scale'] = curriculum.get_noise_amplitude(current_epoch)
        return original_generate_signal(*args, **kwargs)
    
    import src.utils.signal_generator
    src.utils.signal_generator.generate_signal = curriculum_generate_signal

    # training loop
    pbar = tqdm(range(config.training.num_epochs), desc="Training")
    for epoch in range(config.training.num_epochs):
        # Update current epoch for signal generator
        current_epoch = epoch
        
        # Update curriculum learning parameters
        curriculum_stats = curriculum.step(epoch)
        
        signals, targets = generate_batch()
        signals, targets = signals.to(device), targets.to(device)
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
            
        # Update progress bar with curriculum stats
        pbar.set_postfix(
            loss=current_loss,
            best_loss=monitor.best_loss_value,
            patience=monitor.patience_counter,
            **curriculum_stats
        )
        pbar.update(1)
            
    pbar.close()
    
    # Restore original generate_signal function
    src.utils.signal_generator.generate_signal = original_generate_signal
            
    # Save final loss plot
    monitor.save_final_plot()

    # Pass device to tracker for evaluation
    tracker.evaluate_model(model, monitor.best_loss_value, device)
    
    print("\nTraining complete! Check the 'experiments' directory for detailed metrics.")
    print("Best model results are maintained in the 'best_model_results' directory.")


if __name__ == "__main__":
    main() 