import numpy as np

def add_awgn(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to signal.

    Args:
        signal (np.ndarray): Input signal
        snr_db (float): Desired SNR in dB

    Returns:
        noisy_signal (np.ndarray)
    """

    # Signal power
    signal_power = np.mean(signal**2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)

    # Noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal