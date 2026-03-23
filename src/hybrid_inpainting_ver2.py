import numpy as np
from scipy.fftpack import dct, idct


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def gradient_smoothing(signal, weight=0.1):
    smoothed = np.copy(signal)
    smoothed[1:-1] = signal[1:-1] + weight * (
        signal[:-2] + signal[2:] - 2 * signal[1:-1]
    )
    return smoothed


def inpaint_signal(observed_signal, mask,
                   num_iters=100,
                   lambda_sparse=0.2,
                   lambda_grad=0.2):
    """
    Improved Hybrid Inpainting Algorithm
    """

    # 🔹 Normalize
    max_val = np.max(np.abs(observed_signal)) + 1e-8
    x = observed_signal / max_val

    y = np.copy(x)   # for momentum
    t = 1

    for i in range(num_iters):

        # 🔹 Adaptive threshold
        threshold = lambda_sparse * (1 - i / num_iters)

        # 🔹 DCT
        X = dct(y, norm='ortho')

        # 🔹 Sparse coding
        X = soft_threshold(X, threshold)

        # 🔹 IDCT
        x_new = idct(X, norm='ortho')

        # 🔹 Gradient smoothing
        x_new = gradient_smoothing(x_new, weight=lambda_grad)

        # 🔹 Enforce known samples
        x_new = mask * (observed_signal / max_val) + (1 - mask) * x_new

        # 🔹 FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

    # 🔹 Denormalize
    return x * max_val