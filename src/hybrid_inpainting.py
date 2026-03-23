import numpy as np
from scipy.fftpack import dct, idct


def hard_threshold(X, keep_ratio=0.1):
    """
    Keep top K% largest coefficients
    """
    K = int(len(X) * keep_ratio)

    # Get indices of largest K values
    idx = np.argsort(np.abs(X))[-K:]

    X_new = np.zeros_like(X)
    X_new[idx] = X[idx]

    return X_new


def gradient_smoothing(signal, mask, weight=0.1):
    """
    Apply smoothing ONLY on missing samples
    """
    smoothed = np.copy(signal)

    for i in range(1, len(signal) - 1):
        if mask[i] == 0:  # only missing points
            smoothed[i] = signal[i] + weight * (
                signal[i-1] + signal[i+1] - 2 * signal[i]
            )

    return smoothed


def process_frame(frame, mask_frame,
                  num_iters=50,
                  keep_ratio=0.1,
                  lambda_grad=0.1):

    x = np.copy(frame)

    for _ in range(num_iters):

        # DCT
        X = dct(x, norm='ortho')

        # Hard thresholding
        X = hard_threshold(X, keep_ratio)

        # Inverse DCT
        x = idct(X, norm='ortho')

        # Gradient smoothing (only missing)
        x = gradient_smoothing(x, mask_frame, lambda_grad)

        # Enforce known samples
        x = mask_frame * frame + (1 - mask_frame) * x

    return x


def inpaint_signal(observed_signal, mask,
                   frame_size=512,
                   hop_size=256,
                   num_iters=50,
                   keep_ratio=0.1,
                   lambda_grad=0.1):
    """
    Block-based inpainting with IHT
    """

    N = len(observed_signal)

    reconstructed = np.zeros(N)
    weight = np.zeros(N)

    for start in range(0, N - frame_size, hop_size):

        end = start + frame_size

        frame = observed_signal[start:end]
        mask_frame = mask[start:end]

        # Process frame
        rec_frame = process_frame(frame, mask_frame,
                                  num_iters,
                                  keep_ratio,
                                  lambda_grad)

        # Overlap-add
        reconstructed[start:end] += rec_frame
        weight[start:end] += 1

    # Avoid division by zero
    weight[weight == 0] = 1

    reconstructed /= weight

    return reconstructed