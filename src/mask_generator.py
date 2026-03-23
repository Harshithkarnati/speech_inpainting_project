import numpy as np

def generate_mask(signal_length, missing_percentage, seed=None):
    """
    Generate a random mask for missing samples.

    Args:
        signal_length (int): Length of the signal
        missing_percentage (float): Percentage of samples to remove (0–100)
        seed (int, optional): For reproducibility

    Returns:
        mask (np.ndarray): Binary mask (1 = observed, 0 = missing)
    """

    if seed is not None:
        np.random.seed(seed)

    # Total samples to remove
    num_missing = int(signal_length * (missing_percentage / 100.0))

    # Start with all ones
    mask = np.ones(signal_length)

    # Randomly choose indices to remove
    missing_indices = np.random.choice(signal_length, num_missing, replace=False)

    # Set those positions to 0
    mask[missing_indices] = 0

    return mask


def apply_mask(signal, mask):
    """
    Apply mask to signal.

    Args:
        signal (np.ndarray): Original signal
        mask (np.ndarray): Mask array

    Returns:
        observed_signal (np.ndarray)
    """
    return signal * mask
