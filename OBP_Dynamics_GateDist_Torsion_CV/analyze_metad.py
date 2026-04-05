#!/usr/bin/env python3

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer


KB_KJ_PER_MOL_K = 0.00831446261815324
PI_TICKS = [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]
PI_TICKLABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

app = typer.Typer(help="Analyze metadynamics results from run_md.py outputs.")


def _normalize_column(name: str) -> str:
    cleaned = name.strip().strip('"')
    cleaned = cleaned.replace("(kJ/mole)", "kJ_per_mol")
    cleaned = cleaned.replace("(ps)", "ps")
    cleaned = cleaned.replace("(K)", "K")
    cleaned = cleaned.replace("(g/mL)", "g_per_mL")
    cleaned = cleaned.replace("(ns/day)", "ns_per_day")
    cleaned = cleaned.replace("(%)", "percent")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    return cleaned.strip("_").lower()


def _read_plumed_fields(path: Path) -> list[str]:
    for line in path.read_text().splitlines():
        if line.startswith("#! FIELDS"):
            return line.split()[2:]
    raise ValueError(f"Could not find '#! FIELDS' header in {path}")


def _load_plumed_table(path: Path) -> pd.DataFrame:
    fields = _read_plumed_fields(path)
    return pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        names=fields,
        engine="python",
    )


def _load_state_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [_normalize_column(col) for col in df.columns]
    if "progress_percent" in df.columns:
        df["progress_percent"] = (
            df["progress_percent"].astype(str).str.rstrip("%").replace("--", np.nan).astype(float)
        )
    for col in df.columns:
        if col == "time_remaining":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].replace("--", np.nan), errors="ignore")
    return df


def _rolling(series: pd.Series, points: int) -> pd.Series:
    if len(series) < 5:
        return series
    window = max(5, min(points, len(series) // 10 if len(series) >= 50 else len(series)))
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _run_plumed_sum_hills(
    hills_path: Path,
    out_path: Path,
    bins: Iterable[int],
    temperature_k: float,
    project_to: str | None = None,
    grid_min: Iterable[float] | None = None,
    grid_max: Iterable[float] | None = None,
) -> None:
    bin_arg = ",".join(str(int(b)) for b in bins)
    cmd = [
        "plumed",
        "sum_hills",
        "--hills",
        str(hills_path),
        "--mintozero",
        "--bin",
        bin_arg,
        "--outfile",
        str(out_path),
    ]
    if grid_min is not None:
        cmd.extend(["--min", ",".join(f"{float(v):.9f}" for v in grid_min)])
    if grid_max is not None:
        cmd.extend(["--max", ",".join(f"{float(v):.9f}" for v in grid_max)])
    if project_to is not None:
        cmd.extend(["--idw", project_to, "--kt", f"{KB_KJ_PER_MOL_K * temperature_k:.9f}"])
    subprocess.run(cmd, check=True)


def _load_fes_2d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _load_plumed_table(path)
    x = np.sort(df.iloc[:, 0].unique())
    y = np.sort(df.iloc[:, 1].unique())
    z = df.pivot(index=df.columns[1], columns=df.columns[0], values=df.columns[2]).sort_index().sort_index(axis=1)
    return x, y, z.to_numpy()


def _load_fes_1d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = _load_plumed_table(path)
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()


def _parse_pdb_atom_records(path: Path) -> list[dict[str, object]]:
    atoms: list[dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atoms.append(
                    {
                        "atom_name": line[12:16].strip(),
                        "res_name": line[17:20].strip(),
                        "chain_id": line[21].strip(),
                        "res_id": int(line[22:26].strip()),
                        "x": float(line[30:38].strip()),
                        "y": float(line[38:46].strip()),
                        "z": float(line[46:54].strip()),
                    }
                )
            except ValueError:
                continue
    return atoms


def _find_pdb_atom_coords(
    atoms: Sequence[dict[str, object]],
    chain_id: str | None,
    res_id: int,
    atom_name: str,
) -> np.ndarray:
    matches = [
        atom
        for atom in atoms
        if atom["res_id"] == res_id
        and atom["atom_name"] == atom_name
        and (chain_id is None or atom["chain_id"] == chain_id)
    ]
    if not matches:
        raise ValueError(
            f"Could not find atom {atom_name} in residue {res_id}"
            + (f" on chain {chain_id}" if chain_id else "")
        )
    if len(matches) > 1:
        raise ValueError(
            f"Atom {atom_name} in residue {res_id}"
            + (f" on chain {chain_id}" if chain_id else "")
            + " is ambiguous in the PDB."
        )
    match = matches[0]
    return np.array([match["x"], match["y"], match["z"]], dtype=float)


def _gate_distance_from_pdb(
    pdb_path: Path,
    chain1: str | None,
    res1: int,
    atom1: str,
    chain2: str | None,
    res2: int,
    atom2: str,
) -> float:
    atoms = _parse_pdb_atom_records(pdb_path)
    xyz1 = _find_pdb_atom_coords(atoms, chain1, res1, atom1)
    xyz2 = _find_pdb_atom_coords(atoms, chain2, res2, atom2)
    return float(np.linalg.norm(xyz1 - xyz2) / 10.0)


def _collect_reference_pdb_distances(
    run_dir: Path,
    chain1: str | None,
    res1: int,
    atom1: str,
    chain2: str | None,
    res2: int,
    atom2: str,
) -> list[dict[str, object]]:
    candidate_paths: list[Path] = []
    preferred_names = ("aligned_reference.pdb", "fixed.pdb", "solvated.pdb")
    for name in preferred_names:
        candidate_paths.extend(sorted(run_dir.glob(f"**/{name}")))
    candidate_paths.extend(sorted(run_dir.glob("*.pdb")))

    seen: set[Path] = set()
    references: list[dict[str, object]] = []
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            gate_nm = _gate_distance_from_pdb(path, chain1, res1, atom1, chain2, res2, atom2)
        except ValueError:
            continue
        references.append(
            {
                "path": path,
                "label": path.relative_to(run_dir) if path.is_relative_to(run_dir) else path.name,
                "gate_nm": gate_nm,
            }
        )
    return references


def _write_partial_hills(source: Path, out_path: Path, stop_after: int) -> None:
    lines = source.read_text().splitlines()
    header = [line for line in lines if line.startswith("#")]
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    if stop_after < 1 or stop_after > len(body):
        raise ValueError(f"stop_after={stop_after} out of range for {source} with {len(body)} hills")
    out_path.write_text("\n".join(header + body[:stop_after]) + "\n")


def _fes_deltas_1d(paths: list[Path]) -> list[dict[str, float]]:
    deltas: list[dict[str, float]] = []
    if len(paths) < 2:
        return deltas

    prev_x, prev_y = _load_fes_1d(paths[0])
    for current in paths[1:]:
        x, y = _load_fes_1d(current)
        if not np.allclose(prev_x, x):
            raise ValueError(f"1D FES grids do not match: {paths[0]} vs {current}")
        diff = np.abs(y - prev_y)
        deltas.append(
            {
                "from": paths[len(deltas)].name,
                "to": current.name,
                "mad_kj_per_mol": float(np.nanmean(diff)),
                "rms_kj_per_mol": float(np.sqrt(np.nanmean(diff**2))),
                "max_kj_per_mol": float(np.nanmax(diff)),
            }
        )
        prev_x, prev_y = x, y
    return deltas


def _fes_deltas_2d(paths: list[Path]) -> list[dict[str, float]]:
    deltas: list[dict[str, float]] = []
    if len(paths) < 2:
        return deltas

    prev_x, prev_y, prev_z = _load_fes_2d(paths[0])
    for current in paths[1:]:
        x, y, z = _load_fes_2d(current)
        if not np.allclose(prev_x, x) or not np.allclose(prev_y, y):
            raise ValueError(f"2D FES grids do not match: {paths[0]} vs {current}")
        diff = np.abs(z - prev_z)
        finite = np.isfinite(diff)
        deltas.append(
            {
                "from": paths[len(deltas)].name,
                "to": current.name,
                "mad_kj_per_mol": float(np.nanmean(diff[finite])),
                "rms_kj_per_mol": float(np.sqrt(np.nanmean(diff[finite] ** 2))),
                "max_kj_per_mol": float(np.nanmax(diff[finite])),
            }
        )
        prev_x, prev_y, prev_z = x, y, z
    return deltas


def _run_block_analysis(
    hills_path: Path,
    analysis_dir: Path,
    hills_count: int,
    bins: tuple[int, int],
    temperature_k: float,
    num_blocks: int,
    grid_min: tuple[float, float],
    grid_max: tuple[float, float],
) -> tuple[list[Path], list[Path], list[dict[str, float]], list[dict[str, float]]]:
    block_gate_paths: list[Path] = []
    block_2d_paths: list[Path] = []
    if hills_count < 2:
        return block_gate_paths, block_2d_paths, [], []

    num_blocks = max(2, min(num_blocks, hills_count))
    with tempfile.TemporaryDirectory(prefix="metad_blocks_", dir=analysis_dir) as tmpdir:
        tmpdir_path = Path(tmpdir)
        boundaries = np.linspace(1, hills_count, num_blocks, dtype=int)
        for idx, stop_after in enumerate(boundaries):
            partial_hills = tmpdir_path / f"block_{idx}.HILLS"
            _write_partial_hills(hills_path, partial_hills, int(stop_after))

            fes2d_path = analysis_dir / f"bck.{idx}.fes_2d.dat"
            fes_gate_path = analysis_dir / f"bck.{idx}.fes_gate_1d.dat"
            _run_plumed_sum_hills(
                hills_path=partial_hills,
                out_path=fes2d_path,
                bins=bins,
                temperature_k=temperature_k,
                grid_min=grid_min,
                grid_max=grid_max,
            )
            _run_plumed_sum_hills(
                hills_path=partial_hills,
                out_path=fes_gate_path,
                bins=bins,
                temperature_k=temperature_k,
                project_to="gate_dist",
                grid_min=grid_min,
                grid_max=grid_max,
            )
            block_2d_paths.append(fes2d_path)
            block_gate_paths.append(fes_gate_path)

    return (
        block_gate_paths,
        block_2d_paths,
        _fes_deltas_1d(block_gate_paths),
        _fes_deltas_2d(block_2d_paths),
    )


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _cleanup_old_block_outputs(analysis_dir: Path) -> None:
    for pattern in ("bck.*.fes_2d.dat", "bck.*.fes_gate_1d.dat"):
        for path in analysis_dir.glob(pattern):
            path.unlink(missing_ok=True)


def _plot_fes_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colvar: pd.DataFrame,
    out_base: Path,
    max_energy: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    z_plot = np.array(z, copy=True)
    z_min = float(np.nanmin(z_plot))
    z_max = float(np.nanpercentile(z_plot, 99.5))
    if max_energy is not None:
        z_max = min(z_max, max_energy)
        z_plot = np.clip(z_plot, z_min, z_max)
    levels = np.linspace(z_min, z_max, 18)
    mesh = ax.contourf(x, y, z_plot, levels=levels, cmap="viridis")
    ax.contour(x, y, z_plot, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.5)
    ax.scatter(
        colvar["gate_dist"],
        colvar["chi1_res2"],
        s=4,
        c="black",
        alpha=0.08,
        linewidths=0,
        label="Sampled states",
    )
    ax.set_xlabel("Gate CV distance (nm)")
    ax.set_ylabel("Gate-sidechain torsion (rad)")
    ax.set_yticks(PI_TICKS)
    ax.set_yticklabels(PI_TICKLABELS)
    ax.set_title("2D Free Energy Surface")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Free energy (kJ/mol)")
    ax.legend(loc="upper right", frameon=True)
    _save_figure(fig, out_base)


def _build_cosine_fes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    temperature_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_centers = np.asarray(x, dtype=float)
    beta = 1.0 / (KB_KJ_PER_MOL_K * temperature_k)
    theta = np.asarray(y, dtype=float)
    z_shift = np.nanmin(z, axis=0, keepdims=True)
    prob_theta = np.exp(-beta * (z - z_shift))

    groups: dict[float, np.ndarray] = {}
    for idx, angle in enumerate(theta):
        cos_val = round(float(np.cos(angle)), 10)
        if cos_val in groups:
            groups[cos_val] = groups[cos_val] + prob_theta[idx, :]
        else:
            groups[cos_val] = prob_theta[idx, :].copy()

    cos_centers = np.array(sorted(groups.keys()), dtype=float)
    prob_cos = np.vstack([groups[val] for val in cos_centers])

    with np.errstate(divide="ignore"):
        z_cos = -(1.0 / beta) * np.log(prob_cos)
    z_cos -= np.nanmin(z_cos)
    z_cos[~np.isfinite(z_cos)] = np.nan
    return x_centers, cos_centers, z_cos


def _plot_fes_2d_cosine(
    x: np.ndarray,
    y_cos: np.ndarray,
    z_cos: np.ndarray,
    colvar: pd.DataFrame,
    out_base: Path,
    max_energy: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    z_plot = np.array(z_cos, copy=True)
    finite = np.isfinite(z_plot)
    z_min = float(np.nanmin(z_plot))
    z_max = float(np.nanpercentile(z_plot[finite], 99.5))
    if max_energy is not None:
        z_max = min(z_max, max_energy)
        z_plot = np.clip(z_plot, z_min, z_max)
    x_edges = np.empty(len(x) + 1, dtype=float)
    x_edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    x_edges[0] = x[0] - 0.5 * (x[1] - x[0])
    x_edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    y_edges = np.empty(len(y_cos) + 1, dtype=float)
    y_edges[1:-1] = 0.5 * (y_cos[:-1] + y_cos[1:])
    y_edges[0] = -1.0
    y_edges[-1] = 1.0
    mesh = ax.pcolormesh(x_edges, y_edges, z_plot, cmap="viridis", shading="auto", vmin=z_min, vmax=z_max)
    ax.scatter(
        colvar["gate_dist"],
        np.cos(colvar["chi1_res2"]),
        s=4,
        c="black",
        alpha=0.08,
        linewidths=0,
        label="Sampled states",
    )
    ax.set_xlabel("Gate CV distance (nm)")
    ax.set_ylabel(r"cos($\chi_1$)")
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_title(r"2D Free Energy Surface in Gate CV vs cos($\chi_1$)")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Free energy (kJ/mol)")
    ax.legend(loc="upper right", frameon=True)
    _save_figure(fig, out_base)


def _plot_fes_gate(x: np.ndarray, y: np.ndarray, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x, y, color="#0b5cab", lw=2.0)
    ax.set_xlabel("Gate CV distance (nm)")
    ax.set_ylabel("Projected free energy (kJ/mol)")
    ax.set_title("1D Free Energy Profile Along Gate CV")
    ax.grid(alpha=0.25)
    _save_figure(fig, out_base)


def _plot_fes_gate_limited(
    x: np.ndarray,
    y: np.ndarray,
    out_base: Path,
    max_energy: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x, y, color="#0b5cab", lw=2.0)
    ax.set_xlabel("Gate CV distance (nm)")
    ax.set_ylabel("Projected free energy (kJ/mol)")
    ax.set_title("1D Free Energy Profile Along Gate CV")
    if max_energy is not None:
        ax.set_ylim(bottom=float(np.nanmin(y)), top=max_energy)
    ax.grid(alpha=0.25)
    _save_figure(fig, out_base)


def _plot_fes_gate_blocks_with_pdb(
    block_gate_paths: list[Path],
    hills: pd.DataFrame,
    out_base: Path,
    max_energy: float | None,
    pdb_refs: list[dict[str, object]],
) -> None:
    if not block_gate_paths:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    cmap = plt.get_cmap("viridis")
    stop_counts = np.linspace(1, len(hills), len(block_gate_paths), dtype=int)
    plotted_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for idx, (path, stop_after) in enumerate(zip(block_gate_paths, stop_counts, strict=True)):
        x, y = _load_fes_1d(path)
        plotted_curves.append((x, y))
        time_ns = float(hills.iloc[int(stop_after) - 1]["time"]) / 1000.0
        color = cmap(idx / max(1, len(block_gate_paths) - 1))
        alpha = 0.45 + 0.45 * ((idx + 1) / len(block_gate_paths))
        ax.plot(x, y, color=color, lw=1.6, alpha=alpha, label=f"{time_ns:.1f} ns")

    final_x, final_y = plotted_curves[-1]
    ax.plot(final_x, final_y, color="black", lw=2.4, alpha=0.9, label="Final FES")
    ymin = float(np.nanmin(final_y))
    ymax = float(np.nanmax(final_y))
    if max_energy is not None:
        ymax = min(ymax, max_energy)
        ax.set_ylim(bottom=ymin, top=ymax)
    yspan = max(1e-6, ymax - ymin)

    pdb_colors = ("#c53b2c", "#dd8a00", "#7a0177", "#238b45", "#2171b5")
    for idx, ref in enumerate(pdb_refs):
        color = pdb_colors[idx % len(pdb_colors)]
        gate_nm = float(ref["gate_nm"])
        label = str(ref["label"])
        ax.axvline(gate_nm, color=color, lw=1.4, ls="--", alpha=0.85)
        ax.scatter([gate_nm], [ymin + 0.03 * yspan], color=color, s=28, zorder=3)
        ax.text(
            gate_nm,
            ymin + 0.08 * yspan,
            label,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            color=color,
            alpha=0.95,
        )

    ax.set_xlabel("Gate CV distance (nm)")
    ax.set_ylabel("Projected free energy (kJ/mol)")
    ax.set_title("1D Gate FES Block Evolution with PDB References")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True, ncol=2, fontsize=8)
    _save_figure(fig, out_base)


def _plot_diagnostics(
    colvar: pd.DataFrame,
    hills: pd.DataFrame,
    state: pd.DataFrame,
    out_base: Path,
) -> None:
    colvar_time_ns = colvar["time"] / 1000.0
    state_time_ns = state["time_ps"] / 1000.0
    wall_bias = colvar["uwall.bias"] + colvar["lwall.bias"]
    total_bias = colvar["metad.bias"] + wall_bias

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()

    axes[0].plot(colvar_time_ns, colvar["gate_dist"], color="#0b5cab", lw=0.9, alpha=0.7)
    axes[0].plot(colvar_time_ns, _rolling(colvar["gate_dist"], 500), color="#c53b2c", lw=1.8)
    axes[0].set_title("Gate Distance")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Distance (nm)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(colvar_time_ns, colvar["chi1_res2"], color="#238b45", lw=0.8, alpha=0.75)
    axes[1].set_title("Chi1 Torsion")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Angle (rad)")
    axes[1].set_yticks(PI_TICKS)
    axes[1].set_yticklabels(PI_TICKLABELS)
    axes[1].grid(alpha=0.25)

    axes[2].plot(colvar_time_ns, colvar["metad.bias"], lw=1.0, label="MetaD bias")
    axes[2].plot(colvar_time_ns, wall_bias, lw=1.0, label="Wall bias")
    axes[2].plot(colvar_time_ns, total_bias, lw=1.2, label="Total bias")
    axes[2].set_title("Bias Evolution")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Bias (kJ/mol)")
    axes[2].legend(frameon=True)
    axes[2].grid(alpha=0.25)

    axes[3].plot(state_time_ns, state["potential_energy_kj_per_mol"], color="#8c2d04", lw=0.9, alpha=0.6)
    axes[3].plot(
        state_time_ns,
        _rolling(state["potential_energy_kj_per_mol"], 100),
        color="black",
        lw=1.8,
    )
    axes[3].set_title("Potential Energy")
    axes[3].set_xlabel("Time (ns)")
    axes[3].set_ylabel("Potential energy (kJ/mol)")
    axes[3].grid(alpha=0.25)

    axes[4].plot(hills["time"] / 1000.0, hills["height"], color="#6a51a3", lw=0.9)
    axes[4].set_title("Deposited Hill Height")
    axes[4].set_xlabel("Time (ns)")
    axes[4].set_ylabel("Height (kJ/mol)")
    axes[4].grid(alpha=0.25)

    ax_temp = axes[5]
    ax_temp.plot(state_time_ns, state["temperature_k"], color="#e6550d", lw=1.0, label="Temperature")
    ax_temp.set_xlabel("Time (ns)")
    ax_temp.set_ylabel("Temperature (K)", color="#e6550d")
    ax_temp.tick_params(axis="y", labelcolor="#e6550d")
    ax_temp.grid(alpha=0.25)
    ax_temp.set_title("Thermodynamic Stability")

    ax_density = ax_temp.twinx()
    ax_density.plot(state_time_ns, state["density_g_per_ml"], color="#3182bd", lw=1.0, label="Density")
    ax_density.set_ylabel("Density (g/mL)", color="#3182bd")
    ax_density.tick_params(axis="y", labelcolor="#3182bd")

    lines = ax_temp.get_lines() + ax_density.get_lines()
    labels = [line.get_label() for line in lines]
    ax_temp.legend(lines, labels, loc="upper right", frameon=True)

    fig.suptitle("Metadynamics Diagnostics", y=0.995)
    fig.tight_layout()
    _save_figure(fig, out_base)


@app.command()
def main(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory produced by run_md.py"),
    temperature_k: float = typer.Option(300.0, help="Simulation temperature used for kT in the 1D projection"),
    gate_bins: int = typer.Option(160, help="Number of bins for the gate CV"),
    chi_bins: int = typer.Option(160, help="Number of bins for the torsion CV"),
    block_count: int = typer.Option(4, min=2, help="Number of cumulative HILLS blocks for convergence checks"),
    fes_max_kj: Optional[float] = typer.Option(
        None,
        help="Upper energy limit for the visible color scale in the 2D FES and y-axis in the 1D FES.",
    ),
    chain1: Optional[str] = typer.Option(None, help="Chain id for residue 1 used in the gate-distance PDB overlay"),
    res1: int = typer.Option(24, help="Residue id for gate atom 1 in the PDB overlay"), # Change when doing OBP2 analysis!
    atom1: str = typer.Option("CA", help="Atom name for gate atom 1 in the PDB overlay"),
    chain2: Optional[str] = typer.Option(None, help="Chain id for residue 2 used in the gate-distance PDB overlay"),
    res2: int = typer.Option(73, help="Residue id for gate atom 2 in the PDB overlay"), # Change when doing OBP2 analysis!
    atom2: str = typer.Option("CA", help="Atom name for gate atom 2 in the PDB overlay"),
) -> None:
    colvar_path = run_dir / "COLVAR"
    hills_path = run_dir / "HILLS"
    state_path = run_dir / "state.log"
    for path in (colvar_path, hills_path, state_path):
        if not path.exists():
            raise typer.BadParameter(f"Required file is missing: {path}")

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_block_outputs(analysis_dir)

    colvar = _load_plumed_table(colvar_path)
    hills = _load_plumed_table(hills_path)
    state = _load_state_log(state_path)

    fes2d_path = analysis_dir / "fes_2d.dat"
    fes_gate_path = analysis_dir / "fes_gate_1d.dat"

    _run_plumed_sum_hills(
        hills_path=hills_path,
        out_path=fes2d_path,
        bins=(gate_bins, chi_bins),
        temperature_k=temperature_k,
    )
    _run_plumed_sum_hills(
        hills_path=hills_path,
        out_path=fes_gate_path,
        bins=(gate_bins, chi_bins),
        temperature_k=temperature_k,
        project_to="gate_dist",
    )

    x2d, y2d, z2d = _load_fes_2d(fes2d_path)
    x1d, y1d = _load_fes_1d(fes_gate_path)
    xcos, ycos, zcos = _build_cosine_fes(x2d, y2d, z2d, temperature_k)

    _plot_fes_2d(x2d, y2d, z2d, colvar, analysis_dir / "fes_2d", fes_max_kj)
    _plot_fes_2d_cosine(xcos, ycos, zcos, colvar, analysis_dir / "fes_2d_cos_chi1", fes_max_kj)
    _plot_fes_gate_limited(x1d, y1d, analysis_dir / "fes_gate_1d", fes_max_kj)
    _plot_diagnostics(colvar, hills, state, analysis_dir / "diagnostics")
    block_gate_paths, block_2d_paths, gate_deltas, fes2d_deltas = _run_block_analysis(
        hills_path=hills_path,
        analysis_dir=analysis_dir,
        hills_count=len(hills),
        bins=(gate_bins, chi_bins),
        temperature_k=temperature_k,
        num_blocks=block_count,
        grid_min=(float(x2d[0]), float(y2d[0])),
        grid_max=(float(x2d[-1]), float(y2d[-1])),
    )
    pdb_refs = _collect_reference_pdb_distances(run_dir, chain1, res1, atom1, chain2, res2, atom2)
    _plot_fes_gate_blocks_with_pdb(
        block_gate_paths=block_gate_paths,
        hills=hills,
        out_base=analysis_dir / "fes_gate_1d_blocks_pdb_overlay",
        max_energy=fes_max_kj,
        pdb_refs=pdb_refs,
    )

    summary = analysis_dir / "analysis_summary.txt"
    summary_lines = [
        f"Run directory: {run_dir}",
        f"COLVAR points: {len(colvar)}",
        f"HILLS count: {len(hills)}",
        f"State log points: {len(state)}",
        f"Gate distance range (nm): {colvar['gate_dist'].min():.3f} .. {colvar['gate_dist'].max():.3f}",
        f"Chi1 range (rad): {colvar['chi1_res2'].min():.3f} .. {colvar['chi1_res2'].max():.3f}",
        f"Max MetaD bias (kJ/mol): {colvar['metad.bias'].max():.3f}",
        f"Max wall bias (kJ/mol): {(colvar['uwall.bias'] + colvar['lwall.bias']).max():.3f}",
        f"Mean temperature (K): {state['temperature_k'].mean():.2f}",
        f"Mean density (g/mL): {state['density_g_per_ml'].mean():.4f}",
        f"Visible FES cap (kJ/mol): {fes_max_kj if fes_max_kj is not None else 'none'}",
        f"Cosine-view rows: {len(ycos)}",
        f"Cumulative block count: {len(block_gate_paths)}",
        f"PDB reference count: {len(pdb_refs)}",
    ]
    if pdb_refs:
        summary_lines.append("PDB gate-distance references (nm):")
        for ref in pdb_refs:
            summary_lines.append(f"  {ref['label']}: {float(ref['gate_nm']):.3f}")
    if gate_deltas:
        summary_lines.append("1D FES drift between cumulative blocks (kJ/mol):")
        for delta in gate_deltas:
            summary_lines.append(
                f"  {delta['from']} -> {delta['to']}: "
                f"MAD={delta['mad_kj_per_mol']:.3f}, RMS={delta['rms_kj_per_mol']:.3f}, MAX={delta['max_kj_per_mol']:.3f}"
            )
    if fes2d_deltas:
        summary_lines.append("2D FES drift between cumulative blocks (kJ/mol):")
        for delta in fes2d_deltas:
            summary_lines.append(
                f"  {delta['from']} -> {delta['to']}: "
                f"MAD={delta['mad_kj_per_mol']:.3f}, RMS={delta['rms_kj_per_mol']:.3f}, MAX={delta['max_kj_per_mol']:.3f}"
            )
    summary.write_text(
        "\n".join(summary_lines) + "\n"
    )

    typer.echo(f"Wrote analysis to {analysis_dir}")


if __name__ == "__main__":
    app()
