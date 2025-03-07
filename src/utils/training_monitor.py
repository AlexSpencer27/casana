from pathlib import Path
import shutil
import plotly.graph_objects as go
from typing import List, Optional
import numpy as np
import math

from src.config.config import config


class TrainingMonitor:
    """A class to monitor and visualize training progress."""
    
    def __init__(self, project_root: Path, save_frequency: int = 5, base_round: int = 10):
        """Initialize the training monitor.
        
        Args:
            project_root: Root directory of the project
            save_frequency: How often to save plots (in epochs)
            base_round: Base number to round number of points to for MA window calculation
        """
        self.save_dir = project_root / 'experiments' / config.model.name / 'figures' 
        shutil.rmtree(self.save_dir, ignore_errors=True)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_frequency = save_frequency
        self.base_round = base_round
        self.losses: List[float] = []
        
        # Early stopping variables
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.min_delta = config.training.early_stopping.min_delta
        self.patience = config.training.early_stopping.patience
        
        # Create save directory if it doesn't exist
        self.save_dir.mkdir(exist_ok=True)
    
    def _get_ma_window(self) -> int:
        """Calculate moving average window size based on current number of points.
        
        Returns:
            Window size for moving average
        """
        n_points = len(self.losses)
        rounded_points = math.ceil(n_points / self.base_round) 
        ma_window = np.clip(rounded_points, 1, 50) # clip from 1 to 50
        return ma_window
    
    def _calculate_moving_average(self) -> np.ndarray:
        """Calculate moving average of losses.
        
        Returns:
            Array of moving averages with same length as losses
        """
        losses_array = np.array(self.losses)
        ma_window = self._get_ma_window()
        weights = np.ones(ma_window) / ma_window
        ma = np.convolve(losses_array, weights, mode='valid')
        # Pad the beginning to match length
        padding = np.full(ma_window - 1, np.nan)
        return np.concatenate([padding, ma])
    
    def update(self, loss: float, epoch: Optional[int] = None) -> bool:
        """Update the monitor with a new loss value and check early stopping condition.
        
        Args:
            loss: The loss value to record
            epoch: Current epoch number (optional)
            
        Returns:
            bool: True if training should continue, False if early stopping triggered
        """
        self.losses.append(loss)
        
        # Early stopping check
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Save plot if we hit the save frequency
        if epoch is not None and (epoch + 1) % self.save_frequency == 0:
            self.save_plot("loss.png")
            
        # Return True to continue training, False to stop
        return self.patience_counter < self.patience
    
    def save_plot(self, filename: str) -> None:
        """Save the current loss plot to a PNG file.
        
        Args:
            filename: Name of the file to save the plot to
        """
        fig = go.Figure()
        
        # Add individual loss points
        fig.add_trace(go.Scatter(
            y=self.losses,
            mode='markers',
            name='Loss',
            marker=dict(
                size=4,
                color="blue",
                opacity=0.5
            ),
            showlegend=True
        ))
        
        # Add moving average line
        ma_window = self._get_ma_window()
        if len(self.losses) >= ma_window:
            ma = self._calculate_moving_average()
            fig.add_trace(go.Scatter(
                y=ma,
                mode='lines',
                name=f'{ma_window}-point Moving Average',
                line=dict(
                    color="blue",
                    width=2
                ),
                showlegend=True
            ))
        
        fig.update_layout(
            title='Training Loss Over Time',
            xaxis_title='Iteration',
            yaxis_title='Loss',
            yaxis_type='log',
            template='plotly_white',
            showlegend=True,
            width=1000,
            height=600,
            font=dict(size=14),
            # Remove gridlines
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        
        fig.write_image(str(self.save_dir / filename))
    
    def save_final_plot(self) -> None:
        """Save the final training loss plot."""
        self.save_plot("loss.png")
    
    @property
    def current_loss(self) -> float:
        """Get the most recent loss value."""
        return self.losses[-1] if self.losses else float('inf')
    
    @property
    def should_stop(self) -> bool:
        """Check if early stopping should be triggered.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        return self.patience_counter >= self.patience
    
    @property
    def best_loss_value(self) -> float:
        """Get the best loss value seen so far.
        
        Returns:
            float: Best loss value
        """
        return self.best_loss 