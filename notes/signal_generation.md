# Signal Generation Properties

### 1. Two Peaks with Ramps
* Peaks are generated using Hanning windows
* Peak widths are randomly selected between 10-40 samples
* First peak position is randomly chosen between 0.1-0.5 seconds
* Second peak position is randomly chosen between 0.6-1.8 seconds
* Peak amplitudes are fixed (1.0 and 0.8)

### 2. Complex Noise
* Composed of 2-5 random sine waves
* Frequencies range from 1-10 Hz
* Amplitudes range from 0.05-0.2
* Random phase shifts

### 3. Signal Normalization
* The entire signal is normalized (zero mean, unit variance)

---

# Architecture Decisions

### 1. Convolutional Layers
* CNNs excel at pattern recognition in signals
* They can learn to detect the specific peak shapes (Hanning windows) regardless of position
* They're translation-invariant, which helps with detecting peaks at different positions

### 2. Frequency Domain Information
* Since the noise consists of sine waves, processing in the frequency domain might help
* We could incorporate FFT features or spectrograms

### 3. Attention to Signal Scales
* The peaks have specific widths (10-40 samples)
* Multi-scale processing could help identify features at different resolutions

### 4. Pooling Considerations
* We need precise position detection, so aggressive pooling might reduce accuracy
* Consider dilated convolutions or limited pooling to maintain positional information

### 5. Regression Task Specifics
* The output is normalized positions (between 0-1)
* The model should ensure the predicted positions follow the constraint that peak1 < midpoint < peak2