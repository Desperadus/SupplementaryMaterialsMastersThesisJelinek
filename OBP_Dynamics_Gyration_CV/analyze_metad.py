#!/usr/bin/env python3

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer


KB_KJ_PER_MOL_K = 0.00831446261815324

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


def _infer_hills_cv_names(path: Path) -> list[str]:
    fields = _read_plumed_fields(path)
    cv_names: list[str] = []
    reserved = {"time", "height", "biasf"}
    sigma_prefix = "sigma_"
    for field in fields:
        if field in reserved or field.startswith(sigma_prefix):
            continue
        cv_names.append(field)
    if not cv_names:
        raise ValueError(f"Could not infer CV columns from HILLS header in {path}")
    return cv_names


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
    cv_names = _infer_hills_cv_names(hills_path)
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
    if project_to is not None and len(cv_names) > 1:
        cmd.extend(["--idw", project_to, "--kt", f"{KB_KJ_PER_MOL_K * temperature_k:.9f}"])
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not run `plumed sum_hills` because the `plumed` executable was not found on PATH."
        ) from exc


def _load_fes_2d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _load_plumed_table(path)
    x = np.sort(df.iloc[:, 0].unique())
    y = np.sort(df.iloc[:, 1].unique())
    z = df.pivot(index=df.columns[1], columns=df.columns[0], values=df.columns[2]).sort_index().sort_index(axis=1)
    z_values = z.to_numpy(dtype=float)
    z_values[~np.isfinite(z_values)] = np.nan
    return x, y, z_values


def _load_fes_1d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = _load_plumed_table(path)
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()


def _write_partial_hills(source: Path, out_path: Path, stop_after: int) -> None:
    lines = source.read_text().splitlines()
    header = [line for line in lines if line.startswith("#")]
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    if stop_after < 1 or stop_after > len(body):
        raise ValueError(f"stop_after={stop_after} out of range for {source} with {len(body)} hills")
    out_path.write_text("\n".join(header + body[:stop_after]) + "\n")


def _write_partial_colvar(source: Path, out_path: Path, stop_time: float) -> None:
    lines = source.read_text().splitlines()
    header = [line for line in lines if line.startswith("#")]
    body: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        time_value = float(stripped.split()[0])
        if time_value <= stop_time:
            body.append(line)
    if not body:
        raise ValueError(f"No COLVAR samples at or before {stop_time} ps in {source}")
    out_path.write_text("\n".join(header + body) + "\n")


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


def _run_plumed_driver_reweight(
    colvar_path: Path,
    out_path: Path,
    temperature_k: float,
    bins: tuple[int, int],
    grid_min: tuple[float, float],
    grid_max: tuple[float, float],
) -> None:
    colvar_path = colvar_path.resolve()
    out_path = out_path.resolve()
    fields = set(_read_plumed_fields(colvar_path))
    has_rbias = "metad.rbias" in fields
    if "uwall.bias" not in fields or "lwall.bias" not in fields:
        raise ValueError(f"COLVAR file is missing wall bias columns required for reweighting: {colvar_path}")
    if not has_rbias and "metad.bias" not in fields:
        raise ValueError(f"COLVAR file is missing metadynamics bias columns required for reweighting: {colvar_path}")

    bandwidths = []
    for lower, upper, nbin in zip(grid_min, grid_max, bins, strict=True):
        spacing = (float(upper) - float(lower)) / max(1, int(nbin))
        bandwidths.append(max(spacing * 0.5, 1.0e-6))

    metad_field = "metad.rbias" if has_rbias else "metad.bias"
    weight_args = f"{metad_field},uwall.bias,lwall.bias"
    weights_label = "weights"
    script_lines = [
        f'gyr: READ FILE={colvar_path} VALUES=sidechain_gyr IGNORE_TIME',
        f'gate: READ FILE={colvar_path} VALUES=gate_dist IGNORE_TIME',
        f'metad: READ FILE={colvar_path} VALUES={metad_field} IGNORE_TIME',
        f'uwall: READ FILE={colvar_path} VALUES=uwall.bias IGNORE_TIME',
        f'lwall: READ FILE={colvar_path} VALUES=lwall.bias IGNORE_TIME',
        f'{weights_label}: REWEIGHT_BIAS TEMP={temperature_k:.9f} ARG={weight_args}',
        "hist: HISTOGRAM ...",
        "  ARG=gyr,gate",
        f"  GRID_MIN={grid_min[0]:.9f},{grid_min[1]:.9f}",
        f"  GRID_MAX={grid_max[0]:.9f},{grid_max[1]:.9f}",
        f"  GRID_BIN={int(bins[0])},{int(bins[1])}",
        f"  BANDWIDTH={bandwidths[0]:.9f},{bandwidths[1]:.9f}",
        f"  LOGWEIGHTS={weights_label}",
        "... hist:",
        f"fes: CONVERT_TO_FES GRID=hist TEMP={temperature_k:.9f} MINTOZERO",
        f"DUMPGRID GRID=fes FILE={out_path} FMT=%14.9f",
    ]
    with tempfile.TemporaryDirectory(prefix="plumed_reweight_", dir=out_path.parent) as tmpdir:
        script_path = Path(tmpdir) / "plumed_reweight.dat"
        script_path.write_text("\n".join(script_lines) + "\n")
        cmd = [
            "plumed",
            "driver",
            "--plumed",
            str(script_path),
            "--noatoms",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=out_path.parent)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not run `plumed driver` because the `plumed` executable was not found on PATH."
            ) from exc


def _reweighting_mode(colvar_path: Path) -> str:
    fields = set(_read_plumed_fields(colvar_path))
    if "metad.rbias" in fields:
        return "plumed_rbias_plus_walls"
    if "metad.bias" in fields:
        return "legacy_plumed_bias_plus_walls"
    raise ValueError(f"COLVAR file is missing metadynamics bias columns required for reweighting: {colvar_path}")


def _run_block_analysis(
    hills_path: Path,
    colvar_path: Path,
    hills: pd.DataFrame,
    analysis_dir: Path,
    hills_count: int,
    bins: tuple[int, int],
    temperature_k: float,
    num_blocks: int,
    grid_min: tuple[float, float],
    grid_max: tuple[float, float],
) -> tuple[list[Path], list[Path], list[dict[str, float]], list[dict[str, float]]]:
    block_gyr_paths: list[Path] = []
    block_2d_paths: list[Path] = []
    if hills_count < 2:
        return block_gyr_paths, block_2d_paths, [], []

    num_blocks = max(2, min(num_blocks, hills_count))
    with tempfile.TemporaryDirectory(prefix="metad_blocks_", dir=analysis_dir) as tmpdir:
        tmpdir_path = Path(tmpdir)
        boundaries = np.linspace(1, hills_count, num_blocks, dtype=int)
        for idx, stop_after in enumerate(boundaries):
            partial_hills = tmpdir_path / f"block_{idx}.HILLS"
            partial_colvar = tmpdir_path / f"block_{idx}.COLVAR"
            _write_partial_hills(hills_path, partial_hills, int(stop_after))

            fes2d_path = analysis_dir / f"bck.{idx}.fes_2d.dat"
            fes_gyr_path = analysis_dir / f"bck.{idx}.fes_gyr_1d.dat"
            _run_plumed_sum_hills(
                hills_path=partial_hills,
                out_path=fes_gyr_path,
                bins=(bins[0],),
                temperature_k=temperature_k,
                project_to="sidechain_gyr",
                grid_min=(grid_min[0],),
                grid_max=(grid_max[0],),
            )
            stop_time = float(hills.iloc[int(stop_after) - 1]["time"])
            _write_partial_colvar(colvar_path, partial_colvar, stop_time)
            _run_plumed_driver_reweight(
                colvar_path=partial_colvar,
                out_path=fes2d_path,
                temperature_k=temperature_k,
                bins=bins,
                grid_min=grid_min,
                grid_max=grid_max,
            )
            block_2d_paths.append(fes2d_path)
            block_gyr_paths.append(fes_gyr_path)

    return (
        block_gyr_paths,
        block_2d_paths,
        _fes_deltas_1d(block_gyr_paths),
        _fes_deltas_2d(block_2d_paths),
    )


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _cleanup_old_block_outputs(analysis_dir: Path) -> None:
    for pattern in ("bck.*.fes_2d.dat", "bck.*.fes_gyr_1d.dat"):
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
        colvar["sidechain_gyr"],
        colvar["gate_dist"],
        s=4,
        c="black",
        alpha=0.08,
        linewidths=0,
        label="Sampled states",
    )
    ax.set_xlabel("Sidechain gyration radius (nm)")
    ax.set_ylabel("Gate distance (nm)")
    ax.set_title("2D Free Energy Surface: Gyration vs Gate Distance")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Free energy (kJ/mol)")
    ax.legend(loc="upper right", frameon=True)
    _save_figure(fig, out_base)


def _plot_fes_gyr_limited(
    x: np.ndarray,
    y: np.ndarray,
    out_base: Path,
    max_energy: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x, y, color="#0b5cab", lw=2.0)
    ax.set_xlabel("Sidechain gyration radius (nm)")
    ax.set_ylabel("Projected free energy (kJ/mol)")
    ax.set_title("1D Free Energy Profile Along Gyration")
    if max_energy is not None:
        ax.set_ylim(bottom=float(np.nanmin(y)), top=max_energy)
    ax.grid(alpha=0.25)
    _save_figure(fig, out_base)


def _plot_fes_gyr_blocks(
    block_gyr_paths: list[Path],
    hills: pd.DataFrame,
    out_base: Path,
    max_energy: float | None,
) -> None:
    if not block_gyr_paths:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    cmap = plt.get_cmap("viridis")
    stop_counts = np.linspace(1, len(hills), len(block_gyr_paths), dtype=int)
    plotted_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for idx, (path, stop_after) in enumerate(zip(block_gyr_paths, stop_counts, strict=True)):
        x, y = _load_fes_1d(path)
        plotted_curves.append((x, y))
        time_ns = float(hills.iloc[int(stop_after) - 1]["time"]) / 1000.0
        color = cmap(idx / max(1, len(block_gyr_paths) - 1))
        alpha = 0.45 + 0.45 * ((idx + 1) / len(block_gyr_paths))
        ax.plot(x, y, color=color, lw=1.6, alpha=alpha, label=f"{time_ns:.1f} ns")

    final_x, final_y = plotted_curves[-1]
    ax.plot(final_x, final_y, color="black", lw=2.4, alpha=0.9, label="Final FES")
    ymin = float(np.nanmin(final_y))
    ymax = float(np.nanmax(final_y))
    if max_energy is not None:
        ymax = min(ymax, max_energy)
        ax.set_ylim(bottom=ymin, top=ymax)

    ax.set_xlabel("Sidechain gyration radius (nm)")
    ax.set_ylabel("Projected free energy (kJ/mol)")
    ax.set_title("1D Gyration FES Block Evolution")
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

    axes[0].plot(colvar_time_ns, colvar["sidechain_gyr"], color="#0b5cab", lw=0.9, alpha=0.7)
    axes[0].plot(colvar_time_ns, _rolling(colvar["sidechain_gyr"], 500), color="#c53b2c", lw=1.8)
    axes[0].set_title("Sidechain Gyration")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Radius (nm)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(colvar_time_ns, colvar["gate_dist"], color="#238b45", lw=0.8, alpha=0.75)
    axes[1].plot(colvar_time_ns, _rolling(colvar["gate_dist"], 500), color="#111111", lw=1.6, alpha=0.9)
    axes[1].set_title("Gate Distance")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Distance (nm)")
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
    gyr_bins: int = typer.Option(160, help="Number of bins for the gyration CV"),
    gate_bins: int = typer.Option(160, help="Number of bins for the gate distance"),
    block_count: int = typer.Option(4, min=2, help="Number of cumulative HILLS blocks for convergence checks"),
    fes_max_kj: Optional[float] = typer.Option(
        None,
        help="Upper energy limit for the visible color scale in the 2D FES and y-axis in the 1D FES.",
    ),
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
    reweighting_mode = _reweighting_mode(colvar_path)

    fes2d_path = analysis_dir / "fes_2d.dat"
    fes_gyr_path = analysis_dir / "fes_gyr_1d.dat"

    gyr_min = float(min(colvar["sidechain_gyr"].min(), hills["sidechain_gyr"].min()))
    gyr_max = float(max(colvar["sidechain_gyr"].max(), hills["sidechain_gyr"].max()))
    gate_min = float(colvar["gate_dist"].min())
    gate_max = float(colvar["gate_dist"].max())

    _run_plumed_sum_hills(
        hills_path=hills_path,
        out_path=fes_gyr_path,
        bins=(gyr_bins,),
        temperature_k=temperature_k,
        project_to="sidechain_gyr",
        grid_min=(gyr_min,),
        grid_max=(gyr_max,),
    )

    _run_plumed_driver_reweight(
        colvar_path=colvar_path,
        out_path=fes2d_path,
        temperature_k=temperature_k,
        bins=(gyr_bins, gate_bins),
        grid_min=(gyr_min, gate_min),
        grid_max=(gyr_max, gate_max),
    )
    x2d, y2d, z2d = _load_fes_2d(fes2d_path)
    x1d, y1d = _load_fes_1d(fes_gyr_path)

    _plot_fes_2d(x2d, y2d, z2d, colvar, analysis_dir / "fes_2d", fes_max_kj)
    _plot_fes_gyr_limited(x1d, y1d, analysis_dir / "fes_gyr_1d", fes_max_kj)
    _plot_diagnostics(colvar, hills, state, analysis_dir / "diagnostics")
    block_gyr_paths, block_2d_paths, gyr_deltas, fes2d_deltas = _run_block_analysis(
        hills_path=hills_path,
        colvar_path=colvar_path,
        hills=hills,
        analysis_dir=analysis_dir,
        hills_count=len(hills),
        bins=(gyr_bins, gate_bins),
        temperature_k=temperature_k,
        num_blocks=block_count,
        grid_min=(gyr_min, gate_min),
        grid_max=(gyr_max, gate_max),
    )
    _plot_fes_gyr_blocks(
        block_gyr_paths=block_gyr_paths,
        hills=hills,
        out_base=analysis_dir / "fes_gyr_1d_blocks",
        max_energy=fes_max_kj,
    )

    summary = analysis_dir / "analysis_summary.txt"
    summary_lines = [
        f"Run directory: {run_dir}",
        f"COLVAR points: {len(colvar)}",
        f"HILLS count: {len(hills)}",
        f"State log points: {len(state)}",
        f"Gyration range (nm): {colvar['sidechain_gyr'].min():.3f} .. {colvar['sidechain_gyr'].max():.3f}",
        f"Gate distance range (nm): {colvar['gate_dist'].min():.3f} .. {colvar['gate_dist'].max():.3f}",
        f"Max MetaD bias (kJ/mol): {colvar['metad.bias'].max():.3f}",
        f"Max wall bias (kJ/mol): {(colvar['uwall.bias'] + colvar['lwall.bias']).max():.3f}",
        f"Mean temperature (K): {state['temperature_k'].mean():.2f}",
        f"Mean density (g/mL): {state['density_g_per_ml'].mean():.4f}",
        f"Visible FES cap (kJ/mol): {fes_max_kj if fes_max_kj is not None else 'none'}",
        f"2D PLUMED reweighting mode: {reweighting_mode}",
        f"Cumulative block count: {len(block_gyr_paths)}",
    ]
    if gyr_deltas:
        summary_lines.append("1D gyration FES drift between cumulative blocks (kJ/mol):")
        for delta in gyr_deltas:
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
