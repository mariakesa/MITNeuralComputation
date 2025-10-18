from smolagents import tool
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@tool
def plot_all_channels(data_path: str)->None:
    """
    Plot all three-wise 3D projections of neural traces.

    This tool loads 4-channel neural trace data from disk and visualizes
    all combinations of three channels in 3D, colored by total voltage energy.

    Args:
        data_path (str): Path to the data file (.pt or .npy). The file should
            contain a variable "traces" (samples nnels).
    """
    import torch

    # Load data
    if data_path.endswith(".pt"):
        data = torch.load(data_path)
        X = data["traces"].numpy()
    elif data_path.endswith(".npy"):
        X = np.load(data_path)
    else:
        raise ValueError("Only .pt or .npy files are supported.")

    # Center data and prepare combinations
    X_centered = X - X.mean(axis=0, keepdims=True)
    combos = list(itertools.combinations(range(X_centered.shape[1]), 3))
    stride = 10
    X_sampled = X_centered[::stride]

    # Compute energy for coloring
    energy = np.sum(X_sampled**2, axis=1)
    energy = (energy - energy.min()) / (energy.max() - energy.min())

    # Plot all 3D combinations
    fig = plt.figure(figsize=(16, 12))
    for i, (a, b, c) in enumerate(combos, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        p = ax.scatter(
            X_sampled[:, a],
            X_sampled[:, b],
            X_sampled[:, c],
            c=energy,
            cmap="plasma",
            s=2
        )
        ax.set_xlabel(f"Ch{a+1}")
        ax.set_ylabel(f"Ch{b+1}")
        ax.set_zlabel(f"Ch{c+1}")
        ax.set_title(f"Channels {a+1}, {b+1}, {c+1}")
        fig.colorbar(p, ax=ax, shrink=0.6, label="Relative energy")

    plt.suptitle("All three-wise 3D channel projections (colored by voltage energy)", fontsize=14)
    plt.tight_layout()
    plt.show()
