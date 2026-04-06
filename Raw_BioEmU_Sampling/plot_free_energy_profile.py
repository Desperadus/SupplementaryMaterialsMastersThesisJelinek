from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path("ca_distance_res23_res72.dat")
OUTPUT_PNG = Path("free_energy_res23_res72.pdf")
OUTPUT_CSV = Path("free_energy_res23_res72.csv")

TEMPERATURE_K = 300.0
R_KJ_PER_MOL_K = 0.008314462618
N_BINS = 50
EPS = 1e-12


def main() -> None:
    data = np.loadtxt(INPUT_PATH, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    distances = data[:, 1]
    hist, edges = np.histogram(distances, bins=N_BINS, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    free_energy = -R_KJ_PER_MOL_K * TEMPERATURE_K * np.log(hist + EPS)
    finite_mask = hist > 0
    free_energy[finite_mask] -= free_energy[finite_mask].min()
    free_energy[~finite_mask] = np.nan

    np.savetxt(
        OUTPUT_CSV,
        np.column_stack([centers, free_energy, hist]),
        delimiter=",",
        header="distance_angstrom,free_energy_kj_per_mol,probability_density",
        comments="",
    )

    plt.figure(figsize=(7, 4.5))
    plt.plot(centers, free_energy, color="black", lw=2)
    plt.xlabel("Gate-distance (Å)")
    plt.ylabel("Free energy (kJ/mol)")
    plt.title("1D Free Energy Profile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)


if __name__ == "__main__":
    main()
