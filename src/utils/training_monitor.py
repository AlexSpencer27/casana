from pathlib import Path
import shutil
import plotly.graph_objects as go
from typing import List, Optional
import numpy as np
import math
from plotly.subplots import make_subplots
import torch

from src.config.config import config
from src.losses import get_loss
from src.losses.gradient_aware_loss import GradientAwareLoss
from src.losses.simple_mse import SimpleMSELoss

class TrainingMonitor:
    """A class to monitor and visualize training progress."""
    
    def __init__(self, project_root: Path, base_round: int = 10):
        """Initialize the training monitor.
        
        Args:
            project_root: Root directory of the project
            base_round: Base number to round number of points to for MA window calculation
        """
        self.save_dir = project_root / 'experiments' / config.model.name / 'figures' 
        shutil.rmtree(self.save_dir, ignore_errors=True)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.base_round = base_round
        
        # Track total loss
        self.losses: List[float] = []
        
        # Track individual loss components
        self.position_losses: List[float] = []
        self.magnitude_losses: List[float] = []
        self.gradient_losses: List[float] = []
        self.curvature_losses: List[float] = []
        
        # Track curriculum weights
        self.position_weights: List[float] = []
        self.magnitude_weights: List[float] = []
        self.gradient_weights: List[float] = []
        self.curvature_weights: List[float] = []
        
        # Track other curriculum components
        self.learning_rates: List[float] = []
        self.noise_amplitudes: List[float] = []
        
        # Store most recent outputs and targets
        self.recent_outputs: Optional[torch.Tensor] = None
        self.recent_targets: Optional[torch.Tensor] = None
        
        # Early stopping variables
        self.best_loss = float('inf')  # For model saving
        self.best_ma_loss = float('inf')  # For early stopping
        self.patience_counter = 0
        self.min_delta = config.training.early_stopping.min_delta
        self.patience = config.training.early_stopping.patience
        
        # Create save directory if it doesn't exist
        self.save_dir.mkdir(exist_ok=True)
        
        # Get loss function for component computation
        loss_class = get_loss(config.loss.name)
        self.criterion = loss_class()
    
    def _get_ma_window(self) -> int:
        """Calculate moving average window size based on current number of points.
        
        Returns:
            Window size for moving average
        """
        n_points = len(self.losses)
        rounded_points = math.ceil(n_points / self.base_round) 
        ma_window = np.clip(rounded_points, 1, 50) # clip from 1 to 50
        return ma_window
    
    def _calculate_current_ma(self) -> Optional[float]:
        """Calculate moving average for the most recent window.
        
        Returns:
            Current moving average value or None if not enough data points
        """
        if len(self.losses) < 2:
            return None
            
        ma_window = self._get_ma_window()
        if len(self.losses) < ma_window:
            return None
            
        # Calculate MA only for the most recent window
        recent_losses = self.losses[-ma_window:]
        return float(sum(recent_losses) / ma_window)
    
    def _calculate_moving_averages(self, values: List[float]) -> np.ndarray:
        """Calculate moving averages for all points for plotting.
        
        Args:
            values: List of values to calculate moving averages for
            
        Returns:
            Array of moving averages
        """
        ma_window = self._get_ma_window()
        if len(values) < ma_window:
            return np.array([])
            
        # Convert values to numpy array
        values_array = np.array(values)
        
        # Calculate moving averages using convolution
        weights = np.ones(ma_window) / ma_window
        ma = np.convolve(values_array, weights, mode='valid')
        
        # Pad the beginning to match length
        padding = np.full(ma_window - 1, np.nan)
        return np.concatenate([padding, ma])
    
    def update(self, loss: float, epoch: Optional[int] = None, loss_components: Optional[dict] = None, 
              curriculum_weights: Optional[dict] = None, learning_rate: Optional[float] = None,
              noise_amplitude: Optional[float] = None) -> bool:
        """Update the monitor with new values and check early stopping condition.
        
        Args:
            loss: The total loss value to record
            epoch: Current epoch number (optional)
            loss_components: Dictionary containing individual loss components
            curriculum_weights: Dictionary containing current curriculum weights
            learning_rate: Current learning rate
            noise_amplitude: Current noise amplitude
            
        Returns:
            bool: True if training should continue, False if early stopping triggered
        """
        self.losses.append(loss)
        
        # Update loss components if provided
        if loss_components is not None:
            self.position_losses.append(loss_components.get('position', 0.0))
            self.magnitude_losses.append(loss_components.get('magnitude', 0.0))
            self.gradient_losses.append(loss_components.get('gradient', 0.0))
            self.curvature_losses.append(loss_components.get('curvature', 0.0))
            
        # Update curriculum weights if provided
        if curriculum_weights is not None:
            self.position_weights.append(curriculum_weights.get('position', 0.0))
            self.magnitude_weights.append(curriculum_weights.get('magnitude', 0.0))
            self.gradient_weights.append(curriculum_weights.get('gradient', 0.0))
            self.curvature_weights.append(curriculum_weights.get('second_derivative', 0.0))
            
        # Update other curriculum components
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if noise_amplitude is not None:
            self.noise_amplitudes.append(noise_amplitude)
        
        # Update best point estimate loss (for model saving)
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
        
        # Calculate moving average for early stopping decision
        current_ma = self._calculate_current_ma()
        if current_ma is None:
            return True  # Continue training if not enough data
            
        # Simple early stopping based on moving average improvement
        if current_ma < self.best_ma_loss - self.min_delta:
            self.best_ma_loss = current_ma
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Save plot if plot_frequency is set and we hit the frequency
        if epoch is not None and config.visualization.plot_frequency is not None:
            if (epoch + 1) % config.visualization.plot_frequency == 0:
                self.save_plot("loss.png")
            
        # Return True to continue training, False to stop
        return self.patience_counter < self.patience
    
    def save_plot(self, filename: str) -> None:
        """Save the current loss plot to a PNG file.
        
        Args:
            filename: Name of the file to save the plot to
        """
        # Define consistent component order and colors
        COMPONENTS = [
            ('Position', 'rgba(0, 0, 255, 0.75)', 'blue'),  # Blue
            ('Magnitude', 'rgba(255, 0, 0, 0.75)', 'red'),  # Red
            ('Gradient', 'rgba(0, 255, 0, 0.75)', 'green'),  # Green
            ('Curvature', 'rgba(128, 128, 128, 0.75)', 'grey')  # Grey
        ]
        
        # Create figure with 4 subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Total Training Loss',
                'Loss Components',
                'Curriculum Components',
                'Current Predictions vs Targets'
            ),
            vertical_spacing=0.15,  # Reduced spacing
            row_heights=[0.25, 0.25, 0.25, 0.25]  # Equal heights
        )
        
        # Plot 1: Total Loss
        # clip losses to 5th to 95th percentile
        plot_losses = np.clip(self.losses, np.percentile(self.losses, 5), np.percentile(self.losses, 95))
        # Add individual loss points
        fig.add_trace(
            go.Scatter(
                y=plot_losses,
                mode='markers',
                name='Loss',
                marker=dict(
                    size=3,
                    color="blue",
                    opacity=0.8
                ),
                showlegend=True,
                legendgroup="1",
                legendgrouptitle_text="Total Loss"
            ),
            row=1, col=1
        )
        
        # Add moving average line for total loss
        ma_window = self._get_ma_window()
        if len(self.losses) >= ma_window:
            moving_averages = self._calculate_moving_averages(self.losses)
            # Only plot non-NaN values
            valid_indices = ~np.isnan(moving_averages)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(moving_averages))[valid_indices],
                    y=moving_averages[valid_indices],
                    mode='lines',
                    name=f'{ma_window}-point Moving Average',
                    line=dict(
                        color="blue",
                        width=2
                    ),
                    showlegend=True,
                    legendgroup="1"
                ),
                row=1, col=1
            )
        
        # Plot 2: Loss Components as stacked area
        if self.position_losses:  # Only if we have component data
            # Map components to their data
            component_data = [
                (self.position_losses, self.position_weights),
                (self.magnitude_losses, self.magnitude_weights),
                (self.gradient_losses, self.gradient_weights),
                (self.curvature_losses, self.curvature_weights)
            ]
            
            # Calculate weighted moving averages for all components
            ma_values = []
            for (losses, weights), (_, _, _) in zip(component_data, COMPONENTS):
                # Calculate moving average of the weighted loss
                weighted_losses = [loss * weight for loss, weight in zip(losses, weights)]
                moving_avg = self._calculate_moving_averages(weighted_losses)
                valid_indices = ~np.isnan(moving_avg)
                ma_values.append(moving_avg[valid_indices])
            
            x_values = np.arange(len(ma_values[0]))
            cumulative_values = np.zeros_like(ma_values[0])
            
            # Reverse both lists for bottom-to-top stacking
            component_data = component_data[::-1]
            components_rev = COMPONENTS[::-1]
            ma_values = ma_values[::-1]
            
            # Create stacked area plot from bottom to top
            for i, ((name, fill_color, _), values) in enumerate(zip(components_rev, ma_values)):
                # Add current values to cumulative sum
                new_cumulative = cumulative_values + values
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=new_cumulative,
                        mode='lines',
                        fill='tonexty' if i > 0 else 'tozeroy',
                        name=f'{name} Loss',
                        line=dict(width=0),  # Hide the line
                        fillcolor=fill_color,
                        showlegend=True,
                        legendgroup="2",
                        legendgrouptitle_text="Loss Components"
                    ),
                    row=2, col=1
                )
                
                cumulative_values = new_cumulative
        
        # Plot 3: Curriculum Components
        if self.position_weights:  # Only if we have curriculum data
            # Map components to their weight data
            weight_data = [
                (self.position_weights, 'Position'),
                (self.magnitude_weights, 'Magnitude'),
                (self.gradient_weights, 'Gradient'),
                (self.curvature_weights, 'Curvature')
            ]
            
            # Plot loss weights with thicker lines
            for (weights, _), (name, _, line_color) in zip(weight_data, COMPONENTS):
                fig.add_trace(
                    go.Scatter(
                        y=weights,
                        mode='lines',
                        name=f'{name} Weight',
                        line=dict(color=line_color, width=2),  # Increased line width
                        showlegend=True,
                        legendgroup="3",
                        legendgrouptitle_text="Curriculum Components"
                    ),
                    row=3, col=1
                )
            
            # Add learning rate with distinct style
            if self.learning_rates:
                fig.add_trace(
                    go.Scatter(
                        y=self.learning_rates,
                        mode='lines',
                        name='Learning Rate',
                        line=dict(
                            color='orange',
                            width=2,
                            dash='dashdot'  # Changed to dashdot for better visibility
                        ),
                        showlegend=True,
                        legendgroup="3"
                    ),
                    row=3, col=1
                )
            
            # Add noise amplitude with distinct style
            if self.noise_amplitudes:
                fig.add_trace(
                    go.Scatter(
                        y=self.noise_amplitudes,
                        mode='lines',
                        name='Noise Amplitude',
                        line=dict(
                            color='purple',  # Changed to purple for better contrast
                            width=2,
                            dash='dot'
                        ),
                        showlegend=True,
                        legendgroup="3"
                    ),
                    row=3, col=1
                )
            
            # Plot 4: Scatter plot of outputs vs targets
            if self.recent_outputs is not None and self.recent_targets is not None:
                # Extract peak1, midpoint, peak2 from outputs and targets
                peak1_out, midpoint_out, peak2_out = self.recent_outputs[:, 0], self.recent_outputs[:, 1], self.recent_outputs[:, 2]
                peak1_target, midpoint_target, peak2_target = self.recent_targets[:, 0], self.recent_targets[:, 1], self.recent_targets[:, 2]
                
                # Add scatter plots for each component
                fig.add_trace(
                    go.Scatter(
                        x=peak1_target,
                        y=peak1_out,
                        mode='markers',
                        name='Peak 1',
                        marker=dict(color='blue', size=8),
                        showlegend=True,
                        legendgroup="4"
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=midpoint_target,
                        y=midpoint_out,
                        mode='markers',
                        name='Midpoint',
                        marker=dict(color='red', size=8),
                        showlegend=True,
                        legendgroup="4"
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=peak2_target,
                        y=peak2_out,
                        mode='markers',
                        name='Peak 2',
                        marker=dict(color='green', size=8),
                        showlegend=True,
                        legendgroup="4"
                    ),
                    row=4, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Training Progress',
            template='plotly_white',
            showlegend=True,
            width=1200,  # Increased width to accommodate legends
            height=1600,  # Increased height for 4 subplots
            font=dict(size=14),
            # Remove gridlines
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            # Center legends vertically beside each subplot
            legend1=dict(
                yanchor="middle",
                y=0.85,  # Centered with first subplot
                xanchor="left",
                x=1.02
            ),
            legend2=dict(
                yanchor="middle",
                y=0.60,  # Centered with second subplot
                xanchor="left",
                x=1.02
            ),
            legend3=dict(
                yanchor="middle",
                y=0.35,  # Centered with third subplot
                xanchor="left",
                x=1.02
            ),
            legend4=dict(
                yanchor="middle",
                y=0.10,  # Centered with fourth subplot
                xanchor="left",
                x=1.02
            )
        )
        
        # Update axes formatting
        # Update y-axes to log scale for curriculum weights only with better tick formatting
        fig.update_yaxes(
            type="log",
            row=3,
            col=1,
            tickformat=".0e",  # Scientific notation
            dtick=1,  # Log tick spacing
            showgrid=True,  # Show grid for log scale
            gridcolor='rgba(0,0,0,0.1)'  # Light grid
        )
        
        # Update x-axes labels
        fig.update_xaxes(title_text="Iteration", row=3, col=1)
        fig.update_xaxes(title_text="Target Position", row=4, col=1)
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Component Loss", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=1)
        fig.update_yaxes(title_text="Predicted Position", row=4, col=1)
        
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

    def compute_loss_components(self, outputs: torch.Tensor, targets: torch.Tensor, signals: torch.Tensor) -> tuple[dict, dict]:
        """Compute individual loss components and get current curriculum weights.
        
        Args:
            outputs: Model outputs (batch_size, 3)
            targets: Ground truth targets (batch_size, 3)
            signals: Input signals (batch_size, 1, signal_length)
            
        Returns:
            Tuple of (loss_components, curriculum_weights) dictionaries
        """
        # Store recent outputs and targets
        self.recent_outputs = outputs.detach().cpu()
        self.recent_targets = targets.detach().cpu()
        
        with torch.no_grad():
            position_loss, magnitude_loss = self.criterion.compute_position_and_magnitude_losses(outputs, targets, signals)
            # only do if gradient_aware loss
            if isinstance(self.criterion, GradientAwareLoss):
                peak_positions = torch.cat([outputs[:, 0:1], outputs[:, 2:]], dim=1)
                first_deriv, second_deriv = self.criterion.compute_gradients(signals, peak_positions)
                gradient_loss = torch.mean(first_deriv ** 2)
                curvature_loss = -torch.mean(second_deriv)
            else:
                gradient_loss = torch.tensor(0.0)
                curvature_loss = torch.tensor(0.0)
        
        loss_components = {
            'position': position_loss.item(),
            'magnitude': magnitude_loss.item(),
            'gradient': gradient_loss.item(),
            'curvature': curvature_loss.item()
        }
        
        curriculum_weights = {
            'position': config.loss.position_weight,
            'magnitude': config.loss.magnitude_weight,
            'gradient': config.loss.gradient_weight,
            'second_derivative': config.loss.second_derivative_weight
        }
        
        return loss_components, curriculum_weights 