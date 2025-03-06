from pathlib import Path
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import shutil
from src.config.config import config
from src.utils.signal_generator import generate_batch


class ExperimentTracker:
    """Handles model evaluation and experiment tracking."""
    
    def __init__(self, project_root: Path):
        """Initialize the experiment tracker.
        
        Args:
            project_root: Root directory of the project
        """
        self.results_dir = project_root / 'experiments' / config.model.name
        shutil.rmtree(self.results_dir, ignore_errors=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize evaluation criterion
        self.criterion = nn.MSELoss()
    
    def evaluate_model(self, model: nn.Module, final_loss: float) -> None:
        """Evaluate model on test batch and save results.
        
        Args:
            model: The trained model to evaluate
            final_loss: Final training loss
        """
        print("\nEvaluating final predictions...")
        
        # Generate test batch
        signals, targets = generate_batch()
        model.eval()
        
        with torch.no_grad():
            predictions = model(signals)
            eval_loss = self.criterion(predictions, targets).item()
            metrics = self._calculate_metrics(predictions, targets, eval_loss)
            self._print_metrics(metrics)
            self._save_results(model, final_loss, metrics)
            
        return predictions, targets
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, eval_loss: float) -> dict:
        """Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions (batch_size, 3)
            targets: Ground truth values (batch_size, 3)
            eval_loss: MSE loss on evaluation batch
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy
        pred = predictions.detach().cpu().numpy()
        targ = targets.detach().cpu().numpy()
        
        # Calculate average error for each position
        avg_errors = np.mean(np.abs(pred - targ), axis=0)
        position_errors = {
            "peak1_error": float(avg_errors[0]),
            "midpoint_error": float(avg_errors[1]),
            "peak2_error": float(avg_errors[2])
        }
        
        return {
            "eval_loss": eval_loss,
            "position_errors": position_errors
        }
    
    def _print_metrics(self, metrics: dict) -> None:
        """Print metrics in a readable format.
        
        Args:
            metrics: Dictionary of metrics to print
        """
        print("\nFinal Metrics:")
        print(f"Evaluation Loss: {metrics['eval_loss']:.6f}")
        print("\nPosition Errors:")
        for pos, err in metrics["position_errors"].items():
            print(f"  {pos}: {err:.4f}")
    
    def _save_results(self, model: nn.Module, final_loss: float, metrics: dict) -> None:
        """Save experiment results as JSON.
        
        Args:
            model: The trained model
            final_loss: Final training loss
            metrics: Dictionary of computed metrics
        """
        experiment_data = {
            "timestamp": datetime.now().isoformat(),
            "model": {
                "name": config.model.name,
                "num_parameters": model.get_num_parameters()
            },
            "hyperparameters": {
                "num_epochs": config.training.num_epochs,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate
            },
            "training": {
                "final_loss": float(final_loss)
            },
            "evaluation_metrics": metrics
        }
        
        # Generate filename with timestamp
        filename = f"metrics.json"
        
        # Save to JSON file
        with open(self.results_dir / filename, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"\nExperiment results saved to: {filename}") 