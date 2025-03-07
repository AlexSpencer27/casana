from pathlib import Path
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import shutil
from src.config.config import config
from src.utils.signal_generator import generate_batch
from src.utils.plotter import plot_predictions, plot_model_comparison
from src.losses import get_loss


class ExperimentTracker:
    """Handles model evaluation and experiment tracking."""
    
    def __init__(self, project_root: Path):
        """Initialize the experiment tracker.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.results_dir = project_root / 'experiments' / config.model.name
        self.best_results_dir = project_root / 'best_model_results' / config.model.name
        
        # Clear and create current experiment directory
        shutil.rmtree(self.results_dir, ignore_errors=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize evaluation criterion
        loss_class = get_loss(config.loss.name)
        self.criterion = loss_class()
    
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
            eval_loss = self.criterion(predictions, targets, signals).item()
            metrics = self._calculate_metrics(predictions, targets, signals)
            self._print_metrics(metrics)
            
            # Save all results before updating best
            self._save_results(model, final_loss, metrics)
            plot_predictions(self.project_root, signals, targets, predictions)
            
            # Now update best results after all files are created
            self._update_best_results(metrics)
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, signals: torch.Tensor) -> dict:
        """Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions (batch_size, 3)
            targets: Ground truth values (batch_size, 3)
            signals: Input signals (batch_size, 1, signal_length)
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy
        pred = predictions.detach().cpu().numpy()
        targ = targets.detach().cpu().numpy()
        
        # Calculate position errors
        position_errors = np.mean(np.abs(pred - targ), axis=0)
        position_metrics = {
            "peak1_error": float(position_errors[0]),
            "midpoint_error": float(position_errors[1]),
            "peak2_error": float(position_errors[2])
        }
        
        # Calculate magnitude errors
        pred_magnitudes = self.criterion.sample_signal_values(signals, predictions).detach().cpu().numpy()
        target_magnitudes = self.criterion.sample_signal_values(signals, targets).detach().cpu().numpy()
        magnitude_errors = np.mean(np.abs(pred_magnitudes - target_magnitudes), axis=0)
        magnitude_metrics = {
            "peak1_magnitude_error": float(magnitude_errors[0]),
            "midpoint_magnitude_error": float(magnitude_errors[1]),
            "peak2_magnitude_error": float(magnitude_errors[2])
        }
        
        # Calculate combined loss
        eval_loss = self.criterion(predictions, targets, signals).item()
        
        return {
            "eval_loss": eval_loss,
            "position_errors": position_metrics,
            "magnitude_errors": magnitude_metrics
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
            
        print("\nMagnitude Errors:")
        for mag, err in metrics["magnitude_errors"].items():
            print(f"  {mag}: {err:.4f}")
    
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

    def _update_best_results(self, metrics: dict) -> None:
        """Update best results if current experiment is better.
        
        Args:
            metrics: Dictionary of computed metrics
        """
        # Create best results directory if it doesn't exist
        self.best_results_dir.mkdir(exist_ok=True, parents=True)
        
        best_metrics_path = self.best_results_dir / "metrics.json"
        current_eval_loss = metrics["eval_loss"]
        
        # If no previous best exists or current is better, update best results
        should_update = True
        if best_metrics_path.exists():
            with open(best_metrics_path, 'r') as f:
                best_metrics = json.load(f)
                if best_metrics["evaluation_metrics"]["eval_loss"] <= current_eval_loss:
                    should_update = False
        
        if should_update:
            print("\nNew best model found! Updating best results...")
            # Clear previous best results
            shutil.rmtree(self.best_results_dir, ignore_errors=True)
            self.best_results_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy all contents from current results to best results
            shutil.copytree(self.results_dir, self.best_results_dir, dirs_exist_ok=True)
            print(f"Best results updated in: {self.best_results_dir}")
            
            # Update model comparison plot
            plot_model_comparison(self.project_root) 