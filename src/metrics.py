import numpy as np
from pesq import pesq


def _align_signals(original, reconstructed):
    """
    Truncate both signals to a common length and cast to float64.
    """
    min_len = min(len(original), len(reconstructed))
    original = np.asarray(original[:min_len], dtype=np.float64)
    reconstructed = np.asarray(reconstructed[:min_len], dtype=np.float64)
    return original, reconstructed

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
    original, reconstructed = _align_signals(original, reconstructed)

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
    original, reconstructed = _align_signals(original, reconstructed)

    return np.mean((original - reconstructed) ** 2)


def compute_pesq(original, reconstructed, sample_rate):
    """
    Compute PESQ score for speech quality.

    Args:
        original (np.ndarray): Clean reference signal
        reconstructed (np.ndarray): Degraded/reconstructed signal
        sample_rate (int): 8000 (narrowband) or 16000 (wideband)

    Returns:
        float: PESQ score, or np.nan if unsupported/failed.
    """
    original, reconstructed = _align_signals(original, reconstructed)

    if sample_rate not in (8000, 16000):
        return np.nan

    original = np.clip(original, -1.0, 1.0)
    reconstructed = np.clip(reconstructed, -1.0, 1.0)

    mode = "wb" if sample_rate == 16000 else "nb"

    try:
        return float(pesq(sample_rate, original, reconstructed, mode))
    except Exception:
        return np.nan