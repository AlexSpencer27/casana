import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import torch.nn.functional as F

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.losses.gradient_aware_loss import GradientAwareLoss
from src.utils.signal_generator import generate_signal

def normalize_errors(errors):
    """Normalize errors to [0,1] range"""
    errors = np.array(errors)  # Convert list to numpy array
    min_val = np.min(errors)
    max_val = np.max(errors)
    if max_val == min_val:
        return np.zeros_like(errors)
    return (errors - min_val) / (max_val - min_val)

def analyze_loss_components():
    # Initialize loss function
    criterion = GradientAwareLoss()
    
    # Generate a signal with two peaks
    p1_pos = 0.2  # First peak position (normalized)
    midpoint = 0.5  # Midpoint (normalized)
    true_p2_pos = 0.8  # True second peak position
    
    # Generate single signal and reshape to match expected dimensions
    signal = generate_signal(p1_pos, true_p2_pos, add_complex_signal=False)
    signal = (signal - signal.mean()) / signal.std()  # Normalize like in training
    signals = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Create position range for second peak
    num_points = 100
    p2_positions = torch.linspace(midpoint, 0.95, num_points)
    
    # Initialize arrays to store loss components
    position_errors = []
    magnitude_errors = []
    gradient_errors = []
    curvature_errors = []
    
    # True peak positions
    true_peaks = torch.tensor([[p1_pos, midpoint, true_p2_pos]])
    
    # Calculate losses for each potential second peak position
    for p2_pos in p2_positions:
        # Current peak positions
        pred_peaks = torch.tensor([[p1_pos, midpoint, p2_pos]])
        
        # Get position and magnitude losses
        pos_loss, mag_loss = criterion.compute_position_and_magnitude_losses(pred_peaks, true_peaks, signals)
        
        # Get gradient losses
        peak_positions = torch.cat([pred_peaks[:, 0:1], pred_peaks[:, 2:]], dim=1)
        first_deriv, second_deriv = criterion.compute_gradients(signals, peak_positions)
        
        # Store individual loss components
        position_errors.append(pos_loss.item())
        magnitude_errors.append(mag_loss.item())
        gradient_errors.append(first_deriv.pow(2).mean().item())  # Squared gradient error
        
        # Curvature error - more positive = more error
        curvature_errors.append(second_deriv.mean().item())

    # Create vertically stacked subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Position Error',
            'Magnitude Error',
            'Gradient Error (should be zero at peaks)',
            'Curvature Error (more negative is better)'
        ),
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    # Time points for signal plotting
    time_points = np.linspace(0, 1, len(signal))
    
    # Add traces for each subplot
    error_names = ['Position', 'Magnitude', 'Gradient', 'Curvature']
    errors_list = [position_errors, magnitude_errors, gradient_errors, curvature_errors]
    
    # Normalize all error lists
    normalized_errors = [normalize_errors(errors) for errors in errors_list]
    
    for idx, (errors, name) in enumerate(zip(normalized_errors, error_names), 1):
        # Plot original signal
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=signal.numpy(),
                mode='lines',
                name='Signal',
                line=dict(color='black', width=2)
            ),
            row=idx, col=1
        )

        # only plot from 0.5 to 0.95
        error_time = time_points[time_points >= 0.5]

        # Create error values for each time point
        error_signal = signal[time_points >= 0.5]
        error_values = np.interp(error_time, p2_positions.numpy(), errors)

        # Add signal trace with error-colored markers
        fig.add_trace(
            go.Scatter(
                x=error_time,
                y=error_signal,
                mode='markers',
                name=name,
                marker=dict(
                    size=5,
                    color=error_values,
                    colorscale=[
                        [0, 'rgba(0,255,0,1)'],  # Green for low error
                        [0.5, 'rgba(255,165,0,1)'],  # Orange for medium error
                        [1, 'rgba(255,0,0,1)']  # Red for high error
                    ],
                    showscale=False
                )
            ),
            row=idx, col=1
        )
        
        # Add vertical lines for true peak positions
        for pos in [p1_pos, true_p2_pos]:
            fig.add_vline(
                x=pos, 
                line=dict(dash="dash", color="black", width=1),
                row=idx, col=1
            )
    
    # Update layout
    fig.update_layout(
        title='Loss Components Analysis',
        template='plotly_white',
        showlegend=False,
        height=1200,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update axes labels and ranges
    for i in range(1, 5):
        fig.update_xaxes(
            title_text='Position' if i == 4 else None,
            range=[0, 1],
            row=i, col=1
        )
        fig.update_yaxes(
            title_text='Signal Amplitude' if i == 2 else None,
            range=[min(signal.numpy())-0.1, max(signal.numpy())+0.1],
            row=i, col=1
        )
    
    # Save plot
    output_dir = PROJECT_ROOT / 'experiments' / 'loss_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.write_image(str(output_dir / 'loss_components.png'))
    print(f"Analysis complete! Check {output_dir}/loss_components.png for the visualization.")

if __name__ == "__main__":
    analyze_loss_components() 