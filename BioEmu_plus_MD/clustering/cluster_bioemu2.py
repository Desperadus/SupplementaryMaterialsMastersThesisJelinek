import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import scipy.linalg
import typer
from sklearn.cluster import KMeans

app = typer.Typer(add_completion=False)


def compute_switching_function(
    distances: np.ndarray, r_decay: float = 0.90, n: int = 6
) -> np.ndarray:
    return 1.0 / (1.0 + (distances / r_decay) ** n)


def perform_robust_sfa(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    mean = np.mean(data, axis=0)
    data_centered = data - mean

    covariance_matrix = np.cov(data_centered, rowvar=False)

    epsilon = 1e-8
    covariance_matrix += np.eye(covariance_matrix.shape[0]) * epsilon

    time_derivative = data_centered[1:] - data_centered[:-1]
    derivative_covariance = np.cov(time_derivative, rowvar=False)

    # Solve H * W = lambda * C * W
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        derivative_covariance, 
        covariance_matrix
    )

    idx = np.argsort(eigenvalues)
    selected_vectors = eigenvectors[:, idx[:n_components]]

    return np.dot(data_centered, selected_vectors)


def plot_clusters(
    projection: np.ndarray,
    labels: np.ndarray,
    center_indices: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.6,
        s=15,
        edgecolor='none',
        label="Ensemble",
    )

    centers_x = projection[center_indices, 0]
    centers_y = projection[center_indices, 1]

    plt.scatter(
        centers_x,
        centers_y,
        c="red",
        edgecolors="black",
        s=120,
        marker="X",
        linewidths=1.5,
        label="Cluster Centers (Selected)",
    )

    plt.title("Slow Feature Analysis (SFA) Projection\n(Corrected for Orthogonality)")
    plt.xlabel("Slow Feature 1 (Slowest)")
    plt.ylabel("Slow Feature 2 (2nd Slowest)")
    plt.colorbar(scatter, label="Cluster ID")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input PDB structure(s).",
    ),
    output_dir: Path = typer.Option(
        Path("clusters"),
        "--output-dir",
        "-o",
        help="Directory to save the cluster center structures and plots.",
    ),
    n_clusters: int = typer.Option(
        20, help="Number of clusters (K-Means) to generate."
    ),
    n_sfa_features: int = typer.Option(
        2, help="Number of slow features to extract via SFA."
    ),
    r_decay: float = typer.Option(0.90, help="Switching function cutoff (nm)."),
    n_power: int = typer.Option(6, help="Switching function exponent."),
    dry_run: bool = typer.Option(
        False, help="If True, performs calculations but does not write files."
    ),
):
    typer.echo(f"Loading structures from {input_path}...")
    
    input_str = str(input_path)
    if "*" in input_str:
        import glob
        files = sorted(glob.glob(input_str))
        if not files:
            typer.echo("No files found.")
            raise typer.Exit(code=1)
        traj = md.load(files, top=files[0])
    else:
        if not input_path.exists():
            typer.echo("Input file does not exist.")
            raise typer.Exit(code=1)
        traj = md.load(str(input_path))

    typer.echo(f"Loaded {traj.n_frames} frames.")

    typer.echo("Computing Ca-Ca distances...")
    ca_indices = traj.topology.select("name CA")
    if len(ca_indices) == 0:
        typer.echo("No Alpha Carbons found in topology.")
        raise typer.Exit(code=1)

    n_ca = len(ca_indices)
    pairs = []
    pairs = [
        (ca_indices[i], ca_indices[j]) 
        for i in range(n_ca) 
        for j in range(i + 1, n_ca)
    ]
    
    pairs = np.array(pairs)
    distances = md.compute_distances(traj, pairs)

    typer.echo(f"Applying switching function (r_decay={r_decay}, n={n_power})[cite: 300]...")
    contact_features = compute_switching_function(
        distances, r_decay=r_decay, n=n_power
    )

    typer.echo("Filtering contacts (min < 0.4 and max > 0.6)...")
    min_s = np.min(contact_features, axis=0)
    max_s = np.max(contact_features, axis=0)

    mask = (min_s < 0.4) & (max_s > 0.6)
    filtered_features = contact_features[:, mask]

    typer.echo(
        f"Retained {filtered_features.shape[1]} dynamic contacts out of {contact_features.shape[1]}."
    )

    if filtered_features.shape[1] < 2:
        typer.echo(
            "Error: Not enough dynamic features found for SFA. Check input ensemble diversity."
        )
        raise typer.Exit(code=1)

    typer.echo(f"Performing Robust SFA (Generalized Eigenvalue Problem) ...")
    sfa_projection = perform_robust_sfa(filtered_features, n_components=n_sfa_features)

    typer.echo(f"Clustering into {n_clusters} states using K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(sfa_projection)

    dists_to_centers = kmeans.transform(sfa_projection)
    representative_indices = np.argmin(dists_to_centers, axis=0)
    representative_indices = np.sort(np.unique(representative_indices))

    typer.echo(f"Identified {len(representative_indices)} representative structures.")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        if n_sfa_features >= 2:
            typer.echo("Generating corrected cluster plot...")
            plot_path = output_dir / "sfa_clusters_corrected.pdf"
            plot_clusters(
                projection=sfa_projection,
                labels=kmeans.labels_,
                center_indices=representative_indices,
                output_path=plot_path,
            )
            typer.echo(f"Plot saved to {plot_path}")

        typer.echo(f"Saving PDBs to {output_dir}...")
        for i, frame_idx in enumerate(representative_indices):
            save_path = output_dir / f"cluster_center_{i:03d}_frame_{frame_idx}.pdb"
            traj[frame_idx].save_pdb(str(save_path))
        
        typer.echo("Done.")
    else:
        typer.echo("Dry run finished.")

if __name__ == "__main__":
    app()
