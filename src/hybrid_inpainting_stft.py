import numpy as np
from scipy.signal import stft, istft, get_window

# =========================================================
# 🔹 Soft Threshold (Magnitude for STFT)
# =========================================================
def soft_threshold_mag(Z, lam):
    mag = np.abs(Z)
    phase = np.exp(1j * np.angle(Z))
    mag = np.maximum(mag - lam, 0)
    return mag * phase


# =========================================================
# 🔹 Gradient and Divergence (TV)
# =========================================================
def gradient(x):
    return np.diff(x, append=x[-1])

def divergence(g):
    return np.concatenate(([g[0]], g[1:] - g[:-1]))


# =========================================================
# 🔹 ISTA Solver (STFT)
# =========================================================
def hybrid_ista(y, mask,
                num_iters=250,
                lambda_sparse = 0.0005,
                lambda_tv=0.002,
                step_size = 0.015,
                fs=16000,
                n_fft=512,
                hop_length=256):

    x = mask * y
    window = get_window('hann', n_fft)

    for _ in range(num_iters):

        # -----------------------------
        # Data Fidelity Gradient
        # -----------------------------
        grad_data = mask * (x - y)

        # -----------------------------
        # TV Gradient
        # -----------------------------
        eps = 1e-8
        g = gradient(x)
        grad_tv = divergence(g / (np.abs(g) + eps))

        # -----------------------------
        # Gradient Step
        # -----------------------------
        x = x - step_size * (grad_data + lambda_tv * grad_tv)

        # -----------------------------
        # STFT Sparse Coding
        # -----------------------------
        _, _, Zxx = stft(
            x,
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary='zeros'
        )

        Zxx = soft_threshold_mag(Zxx, lambda_sparse)

        _, x_rec = istft(
            Zxx,
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            input_onesided=True
        )

        # Safety fixes
        if len(x_rec) == 0:
            x_rec = np.zeros_like(y)

        if len(x_rec) < len(y):
            x_rec = np.pad(x_rec, (0, len(y) - len(x_rec)))

        x = x_rec[:len(y)]

        # -----------------------------
        # Enforce Known Samples
        # -----------------------------
        x = mask * y + (1 - mask) * x

    return x


# =========================================================
# 🔹 FISTA Solver (STFT)
# =========================================================
def hybrid_fista(y, mask,
                 num_iters=250,
                 lambda_sparse = 0.0005,
                 lambda_tv=0.002,
                 step_size = 0.015,
                 fs=16000,
                 n_fft=512,
                 hop_length=256):

    x = mask * y
    x_old = np.copy(x)
    t = 1
    window = get_window('hann', n_fft)

    for _ in range(num_iters):

        # Momentum
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y_k = x + ((t - 1) / t_new) * (x - x_old)

        # Data Gradient
        grad_data = mask * (y_k - y)

        # TV Gradient
        eps = 1e-8
        g = gradient(y_k)
        grad_tv = divergence(g / (np.abs(g) + eps))

        # Gradient Step
        x_new = y_k - step_size * (grad_data + lambda_tv * grad_tv)

        # STFT Sparse Coding
        _, _, Zxx = stft(
            x_new,
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary='zeros'
        )

        Zxx = soft_threshold_mag(Zxx, lambda_sparse)

        _, x_rec = istft(
            Zxx,
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            input_onesided=True
        )

        if len(x_rec) == 0:
            x_rec = np.zeros_like(y)

        if len(x_rec) < len(y):
            x_rec = np.pad(x_rec, (0, len(y) - len(x_rec)))

        x_new = x_rec[:len(y)]

        # Enforce known samples
        x_new = mask * y + (1 - mask) * x_new

        # Update
        x_old = x
        x = x_new
        t = t_new

    return x


# =========================================================
# 🔹 Block-Based Processing (USE ISTA FIRST)
# =========================================================
def inpaint_signal(y, mask,
                   frame_size=2048,
                   hop_size=1024,
                   num_iters=250,
                   lambda_sparse = 0.0005,
                   lambda_tv=0.002):

    N = len(y)
    reconstructed = np.zeros(N)
    weight = np.zeros(N)

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_size) / frame_size)

    for start in range(0, N - frame_size + 1, hop_size):
        end = start + frame_size

        frame = y[start:end]
        mask_frame = mask[start:end]

        # 🔹 Using ISTA (as requested)
        rec_frame = hybrid_ista(
            frame,
            mask_frame,
            num_iters=num_iters,
            lambda_sparse=lambda_sparse,
            lambda_tv=lambda_tv
        )

        rec_frame *= window
        reconstructed[start:end] += rec_frame
        weight[start:end] += window

    reconstructed /= (weight + 1e-8)

    return reconstructed