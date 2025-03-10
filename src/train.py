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


def get_current_noise_amplitude(epoch: int) -> float:
    """Calculate the current noise amplitude based on curriculum learning schedule."""
    start_amp = config.signal.complex_noise.start_amplitude
    end_amp = config.signal.complex_noise.end_amplitude
    max_epochs = config.signal.complex_noise.epochs_to_max
    
    # Linear interpolation from start to end amplitude
    progress = min(epoch / max_epochs, 1.0)
    current_amplitude = start_amp + (end_amp - start_amp) * progress
    return current_amplitude


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
    
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Monkey patch the generate_signal function to use curriculum learning
    original_generate_signal = generate_signal
    def curriculum_generate_signal(*args, **kwargs):
        # Add current noise amplitude to signal generation
        kwargs['noise_scale'] = get_current_noise_amplitude(epoch)
        return original_generate_signal(*args, **kwargs)
    
    import src.utils.signal_generator
    src.utils.signal_generator.generate_signal = curriculum_generate_signal

    # training loop
    pbar = tqdm(range(config.training.num_epochs), desc="Training")
    for epoch in range(config.training.num_epochs):
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
            
        # Update progress bar with current noise amplitude
        current_amplitude = get_current_noise_amplitude(epoch)
        noise_status = "off" if current_amplitude < config.signal.complex_noise.min_amplitude else f"{current_amplitude:.3f}"
        pbar.set_postfix(
            loss=current_loss,
            best_loss=monitor.best_loss_value,
            patience=monitor.patience_counter,
            noise=noise_status
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