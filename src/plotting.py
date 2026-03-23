import matplotlib.pyplot as plt
import os


def plot_results(x, y, title, xlabel, ylabel, save_path=None):
    """
    Plot results and optionally save figure.

    Args:
        x (list): X-axis values
        y (list): Y-axis values
        title (str): Plot title
        xlabel (str): X label
        ylabel (str): Y label
        save_path (str, optional): Path to save image
    """

    plt.figure(figsize=(8, 5))

    # Plot line
    plt.plot(x, y, marker='o')

    # Labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Grid for better readability
    plt.grid(True)

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    # Show plot
    plt.show()