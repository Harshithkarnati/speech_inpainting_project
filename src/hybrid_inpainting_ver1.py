import numpy as np
from scipy.fftpack import dct, idct


def soft_threshold(x, threshold):
    """
    Soft thresholding for sparsity
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def gradient_smoothing(signal, weight=0.1):
    """
    Simple gradient regularization using smoothing
    """
    smoothed = np.copy(signal)

    for i in range(1, len(signal) - 1):
        smoothed[i] = signal[i] + weight * (signal[i-1] + signal[i+1] - 2 * signal[i])

    return smoothed


def inpaint_signal(observed_signal, mask, 
                   num_iters=50, 
                   lambda_sparse=0.1, 
                   lambda_grad=0.1):
    """
    Hybrid Gradient Regularized + Spectral Sparse Coding

    Args:
        observed_signal (np.ndarray)
        mask (np.ndarray)
        num_iters (int)
        lambda_sparse (float)
        lambda_grad (float)

    Returns:
        reconstructed_signal (np.ndarray)
    """

    # Initialize with observed signal
    x = np.copy(observed_signal)

    for _ in range(num_iters):

        # 🔹 Step 1: Transform to frequency domain
        X = dct(x, norm='ortho')

        # 🔹 Step 2: Apply sparsity (soft threshold)
        X = soft_threshold(X, lambda_sparse)

        # 🔹 Step 3: Back to time domain
        x = idct(X, norm='ortho')

        # 🔹 Step 4: Gradient smoothing
        x = gradient_smoothing(x, weight=lambda_grad)

        # 🔹 Step 5: Enforce known samples
        x = mask * observed_signal + (1 - mask) * x

    return x