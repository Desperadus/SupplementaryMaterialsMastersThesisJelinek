#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from adjustText import adjust_text


def sanitize_filename(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "column"


def pretty_label(name: str) -> str:
    explicit = {
        "experimental_kd_uM_fluo": "Experimental Kd (uM, Fluo)",
        "vina_score": "VINA docking score (kcal/mol)",
        "log10_Kd": "log10(Kd)",
    }
    if name in explicit:
        return explicit[name]

    tokens = name.replace("-", "_").split("_")
    token_map = {
        "kd": "Kd",
        "ki": "Ki",
        "ic50": "IC50",
        "ec50": "EC50",
        "um": "uM",
        "nm": "nM",
        "mm": "mM",
        "ph": "pH",
        "rmsd": "RMSD",
        "vina": "VINA",
        "score": "score",
        "exp": "Experimental",
        "experimental": "Experimental",
        "fluo": "Fluo",
    }
    pretty_tokens = []
    for token in tokens:
        if not token:
            continue
        mapped = token_map.get(token.lower())
        pretty_tokens.append(mapped if mapped else token.capitalize())
    return " ".join(pretty_tokens) if pretty_tokens else name


def detect_label_column(columns: list[str]) -> str | None:
    candidates = ["Compound", "compound", "Ligand", "ligand", "Name", "name"]
    for c in candidates:
        if c in columns:
            return c
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot experimental_kd_uM_fluo against other numeric columns and "
            "annotate Spearman correlation and p-value."
        )
    )
    parser.add_argument(
        "--csv",
        default="../affinity_merged_with_rdkit_descriptors.csv",
        help="Input CSV file path (default: affinity_boltz_merged.csv)",
    )
    parser.add_argument(
        "--target",
        default="Kd (uM) Fluo",
        help="Target column to compare against (default: experimental_kd_uM_fluo)",
    )
    parser.add_argument(
        "--outdir",
        default="affinity_plots",
        help="Output folder for generated plots (default: affinity_plots)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI (default: 200)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Available columns: {list(df.columns)}"
        )

    numeric_df = df.select_dtypes(include="number")
    if args.target not in numeric_df.columns:
        raise ValueError(f"Target column '{args.target}' is not numeric.")

    compare_columns = [c for c in numeric_df.columns if c != args.target]
    if not compare_columns:
        raise ValueError("No other numeric columns found to compare with target.")

    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    label_col = detect_label_column(df.columns.tolist())

    def make_plot(pair: pd.DataFrame, col: str, label: str, title_suffix: str) -> bool:
        if len(pair) < 3:
            print(f"Skipping '{col}' ({label}): fewer than 3 paired non-NaN values.")
            return False

        rho, pval = spearmanr(pair[args.target], pair[col])

        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        highlight_value = 2.60
        colors = []
        for v in pair[args.target]:
            if np.isclose(v, exclude_value, atol=1e-12, rtol=0.0):
                colors.append("red")
            elif np.isclose(v, highlight_value, atol=1e-12, rtol=0.0):
                colors.append("orange")
            else:
                colors.append("blue")
        sns.regplot(
            data=pair,
            x=col,
            y=args.target,
            ax=ax,
            scatter_kws={"facecolors": colors, "s": 45, "alpha": 0.9},
            line_kws={"linewidth": 1.5, "color": "tab:red"},
            ci=None,
        )

        if label_col and label_col in pair.columns:
            texts = []
            for i in range(pair.shape[0]):
                point_label = str(pair[label_col].iloc[i])
                if point_label and point_label.lower() != "nan":
                    texts.append(
                        ax.text(
                            pair[col].iloc[i],
                            pair[args.target].iloc[i],
                            point_label,
                            fontsize=7,
                            alpha=0.8,
                        )
                    )
            if texts and adjust_text is not None:
                adjust_text(
                    texts,
                    arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1},
                )

        x_label = pretty_label(col)
        y_label = pretty_label(args.target)
        ax.set_title(f"{y_label} vs {x_label}{title_suffix}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        stats_text = f"Spearman rho: {rho:.4f}    p-value: {pval:.4g}    n: {len(pair)}"
        fig.text(0.5, 0.02, stats_text, ha="center", va="bottom", fontsize=10)
        fig.subplots_adjust(bottom=0.2, top=0.9)

        filename = (
            f"{sanitize_filename(args.target)}_vs_{sanitize_filename(col)}_{label}.pdf"
        )
        outpath = outdir / filename
        fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outpath}")
        return True

    created = 0
    skipped = 0
    exclude_value = 3.16

    for col in compare_columns:
        cols = [args.target, col] + ([label_col] if label_col else [])
        pair_all = df[cols].dropna(subset=[args.target, col])
        if make_plot(pair_all, col, "all_rows", ""):
            created += 1
        else:
            skipped += 1

        pair_no_316 = pair_all.loc[
            ~np.isclose(pair_all[args.target], exclude_value, atol=1e-12, rtol=0.0)
        ]
        if make_plot(pair_no_316, col, "exclude_kd_3_16", " (excluding Kd=3.16)"):
            created += 1
        else:
            skipped += 1

    print(f"Done. Created {created} plot(s), skipped {skipped} plot variant(s).")


if __name__ == "__main__":
    main()
