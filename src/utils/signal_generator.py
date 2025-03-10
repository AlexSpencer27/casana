import numpy as np
import torch

from src.config.config import config


def generate_signal(
    p1_position: float,
    p2_position: float,
    p1_amplitude: float = 1.0,
    p2_amplitude: float = 0.8,
    add_complex_signal: bool = True,
) -> torch.Tensor:
    """Generate a signal with two peaks and optional complex noise.

    Args:
        p1_position: Position of the first peak in time units
        p2_position: Position of the second peak in time units
        p1_amplitude: Amplitude of the first peak (default: 1.0)
        p2_amplitude: Amplitude of the second peak (default: 0.8)
        add_complex_signal: Whether to add complex noise (default: True)

    Returns:
        torch.Tensor: The generated signal
    """
    length = config.signal.length
    sampling_rate = config.signal.sampling_rate
    
    signal = np.zeros(length)
    
    # Calculate peak indices based on time positions
    peak1_idx = int(p1_position * sampling_rate)
    peak2_idx = int(p2_position * sampling_rate)

    peak1_width = np.random.randint(10, 40)
    peak2_width = np.random.randint(10, 40)

    # Add peaks with ramps
    signal[peak1_idx - peak1_width : peak1_idx + peak1_width] = p1_amplitude * np.hanning(peak1_width * 2)
    signal[peak2_idx - peak2_width : peak2_idx + peak2_width] = p2_amplitude * np.hanning(peak2_width * 2)

    # Add complex noise signal composed of multiple sine waves
    if add_complex_signal:
        complex_signal = np.zeros(length)
        num_sine_waves = np.random.randint(2, 5)

        for _ in range(num_sine_waves):
            frequency = np.random.uniform(1, 10)
            amplitude = np.random.uniform(0.05, 0.2)
            phase = np.random.uniform(0, 2 * np.pi)
            sine_wave = amplitude * np.sin(
                2 * np.pi * frequency * np.linspace(0, length / sampling_rate, length) + phase
            )
            complex_signal += sine_wave

        signal += complex_signal

    return torch.tensor(signal, dtype=torch.float32)


def generate_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of signals with their corresponding targets.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Batch of signals and their targets
    """
    signals = []
    targets = []
    
    length = config.signal.length
    sampling_rate = config.signal.sampling_rate
    batch_size = config.training.batch_size

    for _ in range(batch_size):
        peak1_time = np.random.uniform(0.1, 0.5) 
        peak2_time = np.random.uniform(0.6, 1.8)
        
        signal = generate_signal(
            p1_position=peak1_time,
            p2_position=peak2_time,
            p1_amplitude=1.0,
            p2_amplitude=0.8,
        )
        signal = (signal - signal.mean()) / signal.std()

        # Calculate normalized sample positions (0-1 range)
        peak1_sample = peak1_time * sampling_rate / length
        peak2_sample = peak2_time * sampling_rate / length
        midpoint_sample = (peak1_time + peak2_time) / 2 * sampling_rate / length

        signals.append(signal)
        targets.append(torch.tensor([peak1_sample, midpoint_sample, peak2_sample], dtype=torch.float32))

    return torch.stack(signals).unsqueeze(1), torch.stack(targets) 