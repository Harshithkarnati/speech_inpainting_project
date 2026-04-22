import numpy as np
from scipy.fftpack import dct, idct

# =========================================================
# 🔹 Soft Threshold (L1 Sparsity)
# =========================================================
def soft_threshold(X, lam):
    return np.sign(X) * np.maximum(np.abs(X) - lam, 0)


# =========================================================
# 🔹 Gradient and Divergence (TV Regularization)
# =========================================================
def gradient(x):
    return np.diff(x, append=x[-1])

def divergence(g):
    return np.concatenate(([g[0]], g[1:] - g[:-1]))


# =========================================================
# 🔹 ISTA Solver (TUNED)
# =========================================================
def hybrid_ista(y, mask,
                num_iters=150,
                lambda_sparse=0.0008,
                lambda_tv=0.002,
                step_size=0.02):

    x = mask * y

    for _ in range(num_iters):

        # Data term
        grad_data = mask * (x - y)

        # TV term (reduced smoothing)
        eps = 1e-8
        g = gradient(x)
        grad_tv = divergence(g / (np.abs(g) + eps))

        # Gradient step
        x = x - step_size * (grad_data + lambda_tv * grad_tv)

        # DCT sparse coding
        X = dct(x, norm='ortho')
        X = soft_threshold(X, lambda_sparse)
        x = idct(X, norm='ortho')

        # Enforce known samples
        x = mask * y + (1 - mask) * x

    return x


# =========================================================
# 🔹 FISTA Solver (TUNED)
# =========================================================
def hybrid_fista(y, mask,
                 num_iters=150,
                 lambda_sparse=0.0008,
                 lambda_tv=0.002,
                 step_size=0.02):

    x = mask * y
    x_old = np.copy(x)
    t = 1

    for _ in range(num_iters):

        # Momentum
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y_k = x + ((t - 1) / t_new) * (x - x_old)

        # Data term
        grad_data = mask * (y_k - y)

        # TV term
        eps = 1e-8
        g = gradient(y_k)
        grad_tv = divergence(g / (np.abs(g) + eps))

        # Gradient step
        x_new = y_k - step_size * (grad_data + lambda_tv * grad_tv)

        # DCT sparse coding
        X = dct(x_new, norm='ortho')
        X = soft_threshold(X, lambda_sparse)
        x_new = idct(X, norm='ortho')

        # Enforce known samples
        x_new = mask * y + (1 - mask) * x_new

        # Update
        x_old = x
        x = x_new
        t = t_new

    return x


# =========================================================
# 🔹 Post Processing (VERY IMPORTANT FOR PESQ)
# =========================================================
def post_process(x):
    # Normalize
    x = x / (np.max(np.abs(x)) + 1e-8)

    # Mild smoothing (removes artifacts, keeps clarity)
    x = 0.98 * x + 0.02 * np.roll(x, 1)

    return x


# =========================================================
# 🔹 Block-Based Processing (FULL SIGNAL)
# =========================================================
def inpaint_signal(y, mask,
                   frame_size=1024,
                   hop_size=512,
                   num_iters=150,
                   lambda_sparse=0.0008,
                   lambda_tv=0.002):

    N = len(y)
    reconstructed = np.zeros(N)
    weight = np.zeros(N)

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_size) / frame_size)

    for start in range(0, N - frame_size + 1, hop_size):
        end = start + frame_size

        frame = y[start:end]
        mask_frame = mask[start:end]

        # 🔥 Use FISTA (better for PESQ)
        rec_frame = hybrid_fista(
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

    # 🔥 Post-processing (big PESQ boost)
    reconstructed = post_process(reconstructed)

    return reconstructed