import shutil
import plotly.graph_objects as go
from pathlib import Path
import torch
import json

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


def plot_model_comparison(project_root: Path) -> None:
    """Create a bar plot comparing the performance of all models in best_model_results.
    
    Args:
        project_root: Root directory of the project
    """
    best_results_dir = project_root / 'best_model_results'
    if not best_results_dir.exists():
        print("No best model results found.")
        return
        
    # Collect data from all models
    model_data = []
    
    for model_dir in best_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        metrics_file = model_dir / 'metrics.json'
        if not metrics_file.exists():
            continue
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        model_data.append((
            metrics['model']['name'],
            metrics['training']['final_loss'],
            metrics['evaluation_metrics']['eval_loss']
        ))
    
    if not model_data:
        print("No model metrics found.")
        return
        
    # Sort by evaluation loss (ascending)
    model_data.sort(key=lambda x: x[2])
    
    # Unzip the sorted data
    models, train_losses, eval_losses = zip(*model_data)
        
    # Create grouped bar plot
    fig = go.Figure(data=[
        go.Bar(name='Training Loss', x=models, y=train_losses, marker_color='grey', opacity=0.5),
        go.Bar(name='Evaluation Loss', x=models, y=eval_losses, marker_color='blue', opacity=0.5)
    ])
    
    # Update layout
    fig.update_layout(
        title='Model Performance Comparison (Sorted by Evaluation Loss)',
        xaxis_title='Model',
        yaxis_title='Loss',
        template='plotly_white',
        barmode='group',
        width=1000,
        height=600,
        font=dict(size=14),
        # Remove gridlines
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    # Save figure
    fig.write_image(str(best_results_dir / 'model_comparison.png')) 