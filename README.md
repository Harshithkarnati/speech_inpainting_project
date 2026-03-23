# Speech Inpainting (Audio Restoration) System

A robust signal processing pipeline designed to reconstruct missing or corrupted segments of audio. This system utilizes a hybrid approach combining **Spectral Sparse Coding** (using Discrete Cosine Transform) and **Gradient Regularization** to iteratively "inpaint" lost data—simulating a resilient solution to problems like VoIP packet loss.

## Features
- **Missing Audio Simulation**: Randomly removes specified percentages of audio samples to benchmark robustness.
- **Noise Injection**: Injects Additive White Gaussian Noise (AWGN) to stress-test the algorithm under degraded real-world conditions.
- **Hybrid Algorithm**: Balances frequency-domain sparsity with time-domain smoothing to produce cleaner, artifact-free reconstructions.
- **Objective Evaluation**: Calculates the Signal-to-Noise Ratio (SNR) for algorithmic benchmarking.
- **Automated Plotting**: Generates and saves line graphs mapping Missing Percentage vs. SNR to visualize performance boundaries.

## Repository Structure

```text
speech_inpainting_project/
├── data/
│   ├── input/                 # Place your clean audio here (e.g., speech1.wav)
│   └── output/                # (Optional) Dump directory for reconstructed audio 
├── src/
│   ├── experiment_missing_noise.py # Tests missing samples + AWGN noise
│   ├── experiment_missing_only.py  # Tests pure missing sample packet loss
│   ├── hybrid_inpainting.py        # Core reconstruction algorithm
│   ├── main.py                     # Entry point for running all experiments
│   ├── mask_generator.py           # Drops audio samples using bit masking
│   ├── metrics.py                  # Computes objective metrics (SNR / MSE)
│   ├── noise.py                    # Gaussian noise generator
│   └── plotting.py                 # Exports benchmark data visualizations
├── results/
│   ├── audio/                 # Exported reconstructed audio dumps
│   ├── logs/                  # System run logs
│   └── plots/                 # Exported SNR benchmark comparison graphs
├── .gitignore                 # Excluded git configuration files
├── requirements.txt           # Project PIP dependencies
└── README.md                  # Project documentation
```

## Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment (`venv` or `conda`).

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/speech-inpainting.git
cd speech-inpainting
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your audio:** Place a clean speech `.wav` file into the `data/input/` directory and name it `speech1.wav` (or update the file paths within the `src/experiment_*.py` files).
2. **Run the experiments:** Execute the main orchestrator script directly from the project root directory:

```bash
python src/main.py
```

This will autonomously:
- Process the `speech1.wav` file through various degrading levels of missing data (10% to 90%).
- Evaluate and score the reconstructions using SNR.
- Generate and save visual line graphs depicting the algorithm's performance curve directly into `results/plots/`.

## The Experiments

- **Experiment 1 (Missing Only):** The baseline benchmark evaluating pure packet loss handling in a clean, noiseless acoustic environment.
- **Experiment 2 (Missing + Noise):** A severe stress test combining high rates of sample drops coupled with heavy background ambient noise injections (e.g., AWGN at 20dB SNR).

## License

This project is open-source and available under the [MIT License](LICENSE).
