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
    
    def update(self, loss: float, epoch: Optional[int] = None) -> None:
        """Update the monitor with a new loss value.
        
        Args:
            loss: The loss value to record
            epoch: Current epoch number (optional)
        """
        self.losses.append(loss)
        
        # Save plot if we hit the save frequency
        if epoch is not None and (epoch + 1) % self.save_frequency == 0:
            self.save_plot("latest_loss.png")
    
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
        # we can delete the latest_loss.png file
        (self.save_dir / 'latest_loss.png').unlink()
        self.save_plot("final_loss.png")
    
    @property
    def current_loss(self) -> float:
        """Get the most recent loss value."""
        return self.losses[-1] if self.losses else float('inf') 