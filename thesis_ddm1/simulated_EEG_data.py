# Import modules
import numpy as np
import mne

## 1. Create synthetic EEG dataset
# Define EEG parameters
sfreq = 512  # Sampling frequency (Hz)
n_channels = 64  # Number of EEG channels
n_samples = sfreq * 2  # 2 seconds of data
n_trials = 100  # Number of trials

# Simulate white noise EEG signal
np.random.seed(42)
data = np.random.randn(n_channels, n_samples) * 5e-6  # Microvolt scale

# Create MNE Raw object
info = mne.create_info(ch_names=n_channels, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)

# Plot raw EEG
raw.plot()

## 2. Add ERP components (N200 and CPP)
# Define N200 waveform
times = np.linspace(0, 1, sfreq)  # 1 second duration
N200 = -5e-6 * np.exp(-((times - 0.2) ** 2) / (2 * (0.05 ** 2)))  # Gaussian shape

# Inject N200 into a specific EEG channel
eeg_data_with_n200 = data.copy()
eeg_data_with_n200[20, sfreq:sfreq + sfreq] += N200  # Add to channel 20 (parietal)

# Define CPP as a ramp function
CPP = np.linspace(0, 5e-6, sfreq)  # Linear increase

# Inject CPP into centro-parietal electrodes
for ch in [10, 20, 25]:  # Example centro-parietal channels
    eeg_data_with_n200[ch, sfreq:sfreq + sfreq] += CPP

## 3. Add trial-to-trial variability (linking EEG to DDM parameters)
# Generate drift rates per trial from a normal distribution
n_trials = 100
drift_rates = np.random.normal(0.5, 0.1, n_trials)  # Mean=0.5, SD=0.1

# Shift N200 latency per trial
n200_latencies = 0.2 - (drift_rates * 0.01)  # Small shift in ms

# Generate trial-by-trial N200s
N200_trials = np.array([-5e-6 * np.exp(-((times - lat) ** 2) / (2 * (0.05 ** 2))) for lat in n200_latencies])

# Inject into EEG data
for i in range(n_trials):
    eeg_data_with_n200[20, sfreq + i * sfreq : sfreq + (i + 1) * sfreq] += N200_trials[i]

# Generate trial-by-trial CPP slopes
CPP_slopes = drift_rates * 5e-6  # Scale drift to EEG amplitude
CPP_trials = np.array([np.linspace(0, slope, sfreq) for slope in CPP_slopes])

# Inject into EEG
for i in range(n_trials):
    for ch in [10, 20, 25]:  # Centro-parietal channels
        eeg_data_with_n200[ch, sfreq + i * sfreq : sfreq + (i + 1) * sfreq] += CPP_trials[i]

## 4. Add realistic EEG noise
# Simulate sensor noise and muscle artifacts
sensor_noise = np.random.randn(n_channels, n_samples) * 2e-6
muscle_artifact = np.sin(2 * np.pi * 50 * times) * 1e-6  # 50 Hz noise

# Inject noise into EEG data
eeg_data_with_n200 += sensor_noise
eeg_data_with_n200[5] += muscle_artifact  # Add muscle noise to channel 5

### Convert to MNE epochs (trial-based data)
# Create event markers (stimuli at 1s intervals)
event_times = np.arange(1, n_trials + 1) * sfreq
events = np.column_stack((event_times, np.zeros(n_trials, dtype=int), np.ones(n_trials, dtype=int)))

# Create epochs
epochs = mne.EpochsArray(eeg_data_with_n200.reshape((n_trials, n_channels, n_samples // n_trials)), info)

# Plot an ERP (N200 visible)
epochs.average().plot()

### Validate simulation
# Compute and plot time-frequency analysis
power = mne.time_frequency.tfr_multitaper(epochs, fmin=1, fmax=40, n_cycles=5, time_bandwidth=2)
power.plot([20])  # Plot for channel 20
