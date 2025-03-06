import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# hyper parameters
num_epochs = 2000
batch_size = 32
length = 2048
sampling_rate = 1024
learning_rate = 0.0001
num_predictions = 32


class PeakDetectionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement the model architecture
        self.fc_layer = nn.Linear(2048, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        x = self.fc_layer(x).view(batch_size, -1)
        return x


# Generate a signal with two peaks
def generate_signal(
    p1_position: float,
    p2_position: float,
    p1_amplitude: float,
    p2_amplitude: float,
    length: int,
    sampling_rate: int,
    add_complex_signal: bool = True,
) -> torch.Tensor:
    signal = np.zeros(length)

    peak1_width = np.random.randint(10, 40)
    peak2_width = np.random.randint(10, 40)

    peak1_idx = int(p1_position * sampling_rate)
    peak2_idx = int(p2_position * sampling_rate)

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


def generate_batch(
    length: int,
    sampling_rate: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    signals = []
    targets = []

    for _ in range(batch_size):
        peak1_time = np.random.uniform(0.1, 0.5)
        peak2_time = np.random.uniform(0.6, 1.8)
        signal = generate_signal(
            p1_position=peak1_time,
            p1_amplitude=1.0,
            p2_position=peak2_time,
            p2_amplitude=0.8,
            length=length,
            sampling_rate=sampling_rate,
        )
        signal = (signal - signal.mean()) / signal.std()

        peak1_sample = peak1_time * sampling_rate / length
        peak2_sample = peak2_time * sampling_rate / length
        midpoint_sample = (peak1_time + peak2_time) / 2 * sampling_rate / length

        signals.append(signal)
        targets.append(torch.tensor([peak1_sample, midpoint_sample, peak2_sample], dtype=torch.float32))

    # signals, [peak1, midpoint, peak2]
    return torch.stack(signals).unsqueeze(1), torch.stack(targets)


def main() -> None:
    # define model / training parameters
    model = PeakDetectionNet()
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(num_epochs):
        signals, targets = generate_batch(length=length, sampling_rate=sampling_rate, batch_size=batch_size)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    # predictions
    signals, targets = generate_batch(length=length, sampling_rate=sampling_rate, batch_size=num_predictions)
    model.eval()
    with torch.no_grad():
        predictions = model(signals)

    # plot predictions
    for i in range(num_predictions):
        plt.figure(figsize=(10, 4))
        plt.plot(signals[i].squeeze().numpy(), label="Original Signal")
        target_positions = targets[i].numpy() * length
        predicted_pos = predictions[i].numpy() * length

        plt.axvline(x=target_positions[0], color="g", linestyle="-", label="Peak 1")
        plt.axvline(x=target_positions[1], color="g", linestyle="--", label="Midpoint")
        plt.axvline(x=target_positions[2], color="g", linestyle="-", label="Peak 2")

        plt.axvline(x=predicted_pos[0], color="r", linestyle="-", label="Predicted Peak 1")
        plt.axvline(x=predicted_pos[1], color="r", linestyle="--", label="Predicted Midpoint")
        plt.axvline(x=predicted_pos[2], color="r", linestyle="-", label="Predicted Peak 2")

        plt.title(f"Example {i + 1}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()