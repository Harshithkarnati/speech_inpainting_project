import numpy as np
import librosa
from tqdm import tqdm

from mask_generator import generate_mask, apply_mask
from hybrid_inpainting import inpaint_signal
from metrics import compute_snr
from plotting import plot_results


def run_missing_only():
    # 🔹 Load audio
    signal, sr = librosa.load("data/input/speech1.wav", sr=16000)

    # 🔹 Missing percentages
    missing_rates = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    snr_values = []

    print("Running Missing Samples Experiment...")

    for rate in tqdm(missing_rates):

        # Step 1: Generate mask
        mask = generate_mask(len(signal), rate, seed=42)

        # Step 2: Apply mask
        observed = apply_mask(signal, mask)

        # Step 3: Reconstruct
        reconstructed = inpaint_signal(observed, mask)

        # Step 4: Compute SNR
        snr = compute_snr(signal, reconstructed)

        snr_values.append(snr)

        print(f"Missing: {rate}% | SNR: {snr:.2f} dB")

    # 🔹 Plot results
    plot_results(
        x=missing_rates,
        y=snr_values,
        title="Missing Samples vs SNR",
        xlabel="Missing Percentage (%)",
        ylabel="SNR (dB)",
        save_path="results/plots/missing_only.png"
    )

    print("Experiment 1 Completed!")