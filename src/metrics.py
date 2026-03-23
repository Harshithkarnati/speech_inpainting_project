import numpy as np

def compute_snr(original, reconstructed):
    """
    Compute Signal-to-Noise Ratio (SNR)

    Args:
        original (np.ndarray): Original clean signal
        reconstructed (np.ndarray): Reconstructed signal

    Returns:
        snr (float): SNR in dB
    """

    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # Compute noise (error)
    noise = original - reconstructed

    # Compute power
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def compute_mse(original, reconstructed):
    """
    Mean Squared Error
    """
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    return np.mean((original - reconstructed) ** 2)