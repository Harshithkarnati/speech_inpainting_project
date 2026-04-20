import numpy as np
import librosa
from tqdm import tqdm

from mask_generator import generate_mask, apply_mask
from hybrid_inpainting import inpaint_signal
from metrics import compute_snr, compute_pesq
from plotting import plot_results


def run_missing_only():
    # 🔹 Load audio
    signal, sr = librosa.load("data/input/speech1.wav", sr=16000)

    # 🔹 Missing percentages
    missing_rates = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    snr_values = []
    pesq_values = []

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
        pesq_score = compute_pesq(signal, reconstructed, sr)

        snr_values.append(snr)
        pesq_values.append(pesq_score)

        print(f"Missing: {rate}% | SNR: {snr:.2f} dB | PESQ: {pesq_score:.3f}")

    # 🔹 Plot results
    plot_results(
        x=missing_rates,
        y=snr_values,
        title="Missing Samples vs SNR",
        xlabel="Missing Percentage (%)",
        ylabel="SNR (dB)",
        save_path="results/plots/missing_only.png"
    )

    # 🔹 Plot PESQ after SNR plot
    plot_results(
        x=missing_rates,
        y=pesq_values,
        title="Missing Samples vs PESQ",
        xlabel="Missing Percentage (%)",
        ylabel="PESQ",
        save_path="results/plots/missing_only_pesq.png"
    )

    print("Experiment 1 Completed!")