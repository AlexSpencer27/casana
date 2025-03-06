import shutil
import plotly.graph_objects as go
from pathlib import Path
import torch

from src.config.config import config


def plot_predictions(project_root: Path, signals: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor) -> None:
    """Plot the signals with their target and predicted peak positions.

    Args:
        signals: Batch of signals to plot
        targets: True peak positions
        predictions: Predicted peak positions
    """
    # Create predictions directory if it doesn't exist
    predictions_dir = project_root / "experiments" / config.model.name / "figures" / "predictions"
    shutil.rmtree(predictions_dir, ignore_errors=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for i in range(config.visualization.num_predictions):
        fig = go.Figure()
        
        # Plot original signal
        signal_data = signals[i].squeeze().numpy()
        x_values = list(range(len(signal_data)))
        fig.add_trace(go.Scatter(
            x=x_values,
            y=signal_data,
            mode='lines',
            name='Original Signal',
            line=dict(color='blue')
        ))
        
        # Get target and predicted positions
        target_positions = targets[i].numpy() * config.signal.length
        predicted_pos = predictions[i].numpy() * config.signal.length
        
        # Add target position lines
        for pos, style in zip(target_positions, ['-', '--', '-']):
            fig.add_trace(go.Scatter(
                x=[pos, pos],
                y=[min(signal_data), max(signal_data)],
                mode='lines',
                name=f'Target {"Midpoint" if style == "--" else "Peak"}',
                line=dict(color='green', dash='dash' if style == '--' else 'solid')
            ))
        
        # Add predicted position lines
        for pos, style in zip(predicted_pos, ['-', '--', '-']):
            fig.add_trace(go.Scatter(
                x=[pos, pos],
                y=[min(signal_data), max(signal_data)],
                mode='lines',
                name=f'Predicted {"Midpoint" if style == "--" else "Peak"}',
                line=dict(color='red', dash='dash' if style == '--' else 'solid')
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Prediction Example {i + 1}',
            xaxis_title='Sample',
            yaxis_title='Amplitude',
            template='plotly_white',
            showlegend=True,
            width=1000,
            height=400,
            font=dict(size=14),
            # Remove gridlines
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        
        # Save figure
        fig.write_image(str(predictions_dir / f'prediction_{i+1}.png')) 