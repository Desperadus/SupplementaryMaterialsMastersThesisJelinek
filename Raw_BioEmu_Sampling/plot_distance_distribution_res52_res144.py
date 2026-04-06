from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


INPUT_PATH = Path("ca_distance_res52_res144.dat")
OUTPUT_PNG = Path("distance_distribution_res52_res144.pdf")


def main() -> None:
    data = np.loadtxt(INPUT_PATH, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    distances = data[:, 1]

    plt.figure(figsize=(7, 4.5))
    sns.histplot(
        distances,
        bins=80,
        stat="density",
        color="#2a6f97",
        edgecolor="white",
        alpha=0.55,
    )
    sns.kdeplot(distances, color="#b22222", lw=2)
    plt.xlabel("CA distance: CYS 53 - CYS 145 (Å)")
    plt.ylabel("Density")
    plt.title("Distance Distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)


if __name__ == "__main__":
    main()
