# Dual-Pathway Neural Network Architecture

This architecture processes signals in both time and frequency domains.

### Time Domain Path
* Three convolutional layers with decreasing kernel sizes:
  * Layer 1: 15
  * Layer 2: 9
  * Layer 3: 5
* Max pooling for downsampling
* Captures spatial patterns of peaks directly in the signal

### Frequency Domain Path
* FFT conversion of input signal
* Separate processing for real and imaginary components
* Two convolutional layers to extract frequency patterns
* Helps identify sinusoidal noise components

### Feature Fusion
* Concatenation of time and frequency features
* Fully connected layers for final position regression

---

## Design Rationale

### Why This Architecture?
The design leverages domain knowledge of the input signal:

* **Dual-pathway approach**: 
  * Targets spatial characteristics of Hanning window peaks (time domain)
  * Processes sine wave noise (frequency domain)
* **Variable kernel sizes**: Accommodates random peak widths (10-40 samples)
* **Adaptive pooling**: Ensures consistent feature dimensions regardless of signal variations
* **Dropout layers**: Prevents overfitting on noise patterns

### Potential Limitations
* **Computational complexity**: FFT processing adds overhead
* **Fixed architecture**: May struggle with extreme peak widths outside training distribution
* **Constraint handling**: No explicit constraint ensuring peak1 < midpoint < peak2
* **Temporal precision**: Pooling operations reduce position precision
* **Basic frequency processing**: More sophisticated spectral processing might improve performance