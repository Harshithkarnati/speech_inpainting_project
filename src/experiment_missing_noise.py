import numpy as np
import librosa
from tqdm import tqdm

from mask_generator import generate_mask, apply_mask
from noise import add_awgn
from hybrid_inpainting import inpaint_signal
from metrics import compute_snr
import matplotlib.pyplot as plt
import os


def run_missing_noise():
    # 🔹 Load audio
    signal, sr = librosa.load("data/input/speech1.wav", sr=16000)

    # 🔹 Missing percentages
    missing_rates = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # 🔹 Multiple noise levels
    snr_noise_levels = [30, 20, 10]

    print("Running Missing + Multiple Noise Experiment...")

    plt.figure(figsize=(8, 5))

    for noise_db in snr_noise_levels:

        snr_values = []

        print(f"\n--- Noise Level: {noise_db} dB ---")

        for rate in tqdm(missing_rates):

            # Step 1: Mask
            mask = generate_mask(len(signal), rate, seed=42)

            # Step 2: Apply mask
            observed = apply_mask(signal, mask)

            # Step 3: Add noise
            noisy_observed = add_awgn(observed, noise_db)

            # Step 4: Reconstruct
            reconstructed = inpaint_signal(noisy_observed, mask)

            # Step 5: Compute SNR
            snr = compute_snr(signal, reconstructed)
            snr_values.append(snr)

            print(f"Missing: {rate}% | Noise: {noise_db} dB | SNR: {snr:.2f} dB")

        # 🔹 Plot this curve
        plt.plot(missing_rates, snr_values, marker='o', label=f"Noise {noise_db} dB")

    # 🔹 Final plot settings
    plt.title("Missing % vs SNR (Multiple Noise Levels)")
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("SNR (dB)")
    plt.grid(True)
    plt.legend()

    # 🔹 Save plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/multiple_noise.png")

    plt.show()

    print("\nExperiment Completed!")