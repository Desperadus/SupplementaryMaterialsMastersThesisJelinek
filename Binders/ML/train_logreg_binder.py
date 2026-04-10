#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None


INPUT_CSV = Path("../affinity_merged_with_rdkit_descriptors.csv")
OUTPUT_DIR = Path("ml_logreg_outputs")

# Global run configuration. Adjust these values here
N_SPLITS = 10
TEST_BINDERS = 5
TEST_NONBINDERS = 5
RANDOM_SEED = 42
FEATURE_NAMES = ["sasa", "logp"] # If you wish to replicate my results with iPTM or Vina score - add those columns here
ID_COLUMN = "Compound"
TARGET_COLUMN = "Kd (uM) Fluo"
TARGET_THRESHOLD = 2.0
PREDICTION_THRESHOLD = 0.5
TARGET_DISPLAY_NAME = "Kd"
BINDER_COLUMN = "is_binder"
PRED_PROB_COLUMN = "pred_prob_binder"
PRED_LABEL_COLUMN = "pred_label"
CORRECT_COLUMN = "correct"
NEGATIVE_CLASS_NAME = "non-binder"
POSITIVE_CLASS_NAME = "binder"
CLASSIFICATION_TARGET_NAMES = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]
TRUE_POSITIVE_LABEL = f"True {POSITIVE_CLASS_NAME}"
TRUE_NEGATIVE_LABEL = f"True {NEGATIVE_CLASS_NAME}"
PRED_POSITIVE_LABEL = f"Pred {POSITIVE_CLASS_NAME}"
PRED_NEGATIVE_LABEL = f"Pred {NEGATIVE_CLASS_NAME}"
TARGET_AXIS_LABEL = f"Experimental {TARGET_DISPLAY_NAME} (uM)"
MODEL_MAX_ITER = 2000



def sanitize_filename(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "plot"


def add_point_labels(ax: plt.Axes, plot_df: pd.DataFrame, *, compact: bool = False) -> None:
    """Add less-cluttered labels, with stronger spreading for non-binders."""
    nonbinder_df = plot_df[plot_df[BINDER_COLUMN] == 0].sort_values(
        by=[TARGET_COLUMN, PRED_PROB_COLUMN],
        ascending=[False, True],
    )
    binder_df = plot_df[plot_df[BINDER_COLUMN] == 1].sort_values(
        by=[TARGET_COLUMN, PRED_PROB_COLUMN],
        ascending=[True, True],
    )

    nonbinder_texts = []
    nonbinder_x = []
    nonbinder_y = []
    for rank, (_, row) in enumerate(nonbinder_df.iterrows()):
        label = f"{row[ID_COLUMN]}" if compact else (
            f"{row[ID_COLUMN]}\n"
            f"{TARGET_DISPLAY_NAME}={row[TARGET_COLUMN]:.2f} uM, p={row[PRED_PROB_COLUMN]:.2f}"
        )
        x0 = float(row[PRED_PROB_COLUMN])
        y0 = float(row[TARGET_COLUMN])
        x_off = 0.008 * ((rank % 5) - 2) if compact else 0.012 * ((rank % 5) - 2)
        y_off = (0.015 + 0.030 * (rank % 4)) if compact else (0.030 + 0.060 * (rank % 4))
        txt = ax.text(x0 + x_off, y0 + y_off, label, fontsize=5 if compact else 7, alpha=0.85)
        nonbinder_texts.append(txt)
        nonbinder_x.append(x0)
        nonbinder_y.append(y0)

    binder_texts = []
    binder_x = []
    binder_y = []
    for rank, (_, row) in enumerate(binder_df.iterrows()):
        label = f"{row[ID_COLUMN]}" if compact else (
            f"{row[ID_COLUMN]}\n"
            f"{TARGET_DISPLAY_NAME}={row[TARGET_COLUMN]:.2f} uM, p={row[PRED_PROB_COLUMN]:.2f}"
        )
        x0 = float(row[PRED_PROB_COLUMN])
        y0 = float(row[TARGET_COLUMN])
        x_off = 0.006 * ((rank % 3) - 1) if compact else 0.008 * ((rank % 3) - 1)
        y_off = (-0.010 if (rank % 2 == 0) else 0.010) if compact else (-0.020 if (rank % 2 == 0) else 0.020)
        txt = ax.text(x0 + x_off, y0 + y_off, label, fontsize=5 if compact else 7, alpha=0.85)
        binder_texts.append(txt)
        binder_x.append(x0)
        binder_y.append(y0)

    if adjust_text is not None and (nonbinder_texts or binder_texts):
        # Some adjustText versions print iterative coordinates to stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            if nonbinder_texts:
                adjust_text(
                    nonbinder_texts,
                    x=nonbinder_x,
                    y=nonbinder_y,
                    only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"},
                    force_text=(0.25, 0.85),
                    force_static=(0.25, 0.85),
                    expand=(1.15, 1.40),
                    ensure_inside_axes=True,
                    arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1},
                    ax=ax,
                )
            if binder_texts:
                adjust_text(
                    binder_texts,
                    x=binder_x,
                    y=binder_y,
                    force_text=(0.12, 0.35),
                    force_static=(0.12, 0.35),
                    expand=(1.05, 1.10),
                    ensure_inside_axes=True,
                    arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1},
                    ax=ax,
                )


def make_split_separation_plot(
    split_id: int,
    split_df: pd.DataFrame,
    split_metrics: dict,
    outdir: Path,
    dataset_name: str,
    file_tag: str,
) -> None:
    """Plot split separation for one split with labels and prediction values."""
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    colors = np.where(split_df[BINDER_COLUMN] == 1, "tab:blue", "tab:orange")
    markers = np.where(split_df[PRED_LABEL_COLUMN] == 1, "o", "s")

    for idx in range(split_df.shape[0]):
        row = split_df.iloc[idx]
        ax.scatter(
            row[PRED_PROB_COLUMN],
            row[TARGET_COLUMN],
            color=colors[idx],
            marker=markers[idx],
            s=70,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )

    add_point_labels(ax, split_df)

    ax.axvline(
        PREDICTION_THRESHOLD,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"p={PREDICTION_THRESHOLD:g} threshold",
    )
    ax.axhline(
        TARGET_THRESHOLD,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"{TARGET_DISPLAY_NAME}={TARGET_THRESHOLD:g} uM binder cutoff",
    )

    ax.set_title(f"Split {split_id:02d}: {dataset_name} Separation (True color, Pred marker)")
    ax.set_xlabel("Predicted probability of binder")
    ax.set_ylabel(TARGET_AXIS_LABEL)

    stats = (
        f"acc={split_metrics['accuracy']:.2f}  prec={split_metrics['precision']:.2f}  "
        f"rec={split_metrics['recall']:.2f}  f1={split_metrics['f1']:.2f}  "
        f"auc={split_metrics['roc_auc']:.2f}"
    )
    fig.text(0.5, 0.02, stats, ha="center", va="bottom", fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color="w", label=TRUE_POSITIVE_LABEL,
                   markerfacecolor="tab:blue", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="w", label=TRUE_NEGATIVE_LABEL,
                   markerfacecolor="tab:orange", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="black", label=PRED_POSITIVE_LABEL, markersize=8),
        plt.Line2D([0], [0], marker="s", linestyle="None", color="black", label=PRED_NEGATIVE_LABEL, markersize=8),
    ]
    ax.legend(handles=handles + ax.lines, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)

    fig.subplots_adjust(bottom=0.2, top=0.9, right=0.80)
    outpath = outdir / f"split_{split_id:02d}_{sanitize_filename(file_tag)}.pdf"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_logreg_formula(clf: Pipeline, split_id: int) -> None:
    """Print logistic regression equation in scaled and raw feature space."""
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["logreg"]

    coef_scaled = logreg.coef_[0]
    intercept_scaled = float(logreg.intercept_[0])
    means = scaler.mean_
    scales = scaler.scale_

    coef_raw = coef_scaled / scales
    intercept_raw = intercept_scaled - np.sum((coef_scaled * means) / scales)

    scaled_terms = " + ".join(
        f"({coef_scaled[j]:+.6f})*(({FEATURE_NAMES[j]}-{means[j]:.6f})/{scales[j]:.6f})"
        for j in range(len(FEATURE_NAMES))
    )
    raw_terms = " + ".join(
        f"({coef_raw[j]:+.6f})*{FEATURE_NAMES[j]}" for j in range(len(FEATURE_NAMES))
    )

    print(f"LogReg formula (Split {split_id:02d}, scaled features):")
    print(f"  p(binder) = sigmoid({intercept_scaled:+.6f} + {scaled_terms})")
    print("Equivalent formula in raw features:")
    print(f"  p(binder) = sigmoid({intercept_raw:+.6f} + {raw_terms})")
    print("  where sigmoid(t) = 1 / (1 + exp(-t))")


def get_logreg_raw_params(clf: Pipeline) -> tuple[float, np.ndarray]:
    """Return intercept and coefficients in raw (unscaled) feature space."""
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["logreg"]

    coef_scaled = logreg.coef_[0]
    intercept_scaled = float(logreg.intercept_[0])
    means = scaler.mean_
    scales = scaler.scale_

    coef_raw = coef_scaled / scales
    intercept_raw = intercept_scaled - np.sum((coef_scaled * means) / scales)
    return intercept_raw, coef_raw


def make_global_separation_plot(
    df_all: pd.DataFrame,
    metrics: dict,
    outdir: Path,
) -> None:
    """Plot all-data separation using averaged model parameters."""
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    colors = np.where(df_all[BINDER_COLUMN] == 1, "tab:blue", "tab:orange")
    markers = np.where(df_all[PRED_LABEL_COLUMN] == 1, "o", "s")

    for idx in range(df_all.shape[0]):
        row = df_all.iloc[idx]
        ax.scatter(
            row[PRED_PROB_COLUMN],
            row[TARGET_COLUMN],
            color=colors[idx],
            marker=markers[idx],
            s=70,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )

    add_point_labels(ax, df_all)

    ax.axvline(
        PREDICTION_THRESHOLD,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"p={PREDICTION_THRESHOLD:g} threshold",
    )
    ax.axhline(
        TARGET_THRESHOLD,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"{TARGET_DISPLAY_NAME}={TARGET_THRESHOLD:g} uM binder cutoff",
    )
    ax.set_title("All Data: Separation (Averaged Model Params)")
    ax.set_xlabel("Predicted probability of binder")
    ax.set_ylabel(TARGET_AXIS_LABEL)

    stats = (
        f"acc={metrics['accuracy']:.2f}  prec={metrics['precision']:.2f}  "
        f"rec={metrics['recall']:.2f}  f1={metrics['f1']:.2f}  "
        f"auc={metrics['roc_auc']:.2f}"
    )
    fig.text(0.5, 0.02, stats, ha="center", va="bottom", fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color="w", label=TRUE_POSITIVE_LABEL,
                   markerfacecolor="tab:blue", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="w", label=TRUE_NEGATIVE_LABEL,
                   markerfacecolor="tab:orange", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="black", label=PRED_POSITIVE_LABEL, markersize=8),
        plt.Line2D([0], [0], marker="s", linestyle="None", color="black", label=PRED_NEGATIVE_LABEL, markersize=8),
    ]
    ax.legend(handles=handles + ax.lines, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)

    fig.subplots_adjust(bottom=0.2, top=0.9, right=0.80)
    outpath = outdir / "all_data_separation_avg_model.pdf"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_feature_importance_plot(
    raw_coefs: list[np.ndarray],
    outdir: Path,
) -> None:
    """Plot mean raw logistic-regression coefficients across splits."""
    coef_matrix = np.vstack(raw_coefs)
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "mean_coef": coef_matrix.mean(axis=0),
            "std_coef": coef_matrix.std(axis=0),
        }
    ).sort_values("mean_coef", key=lambda s: s.abs(), ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = np.where(importance_df["mean_coef"] >= 0, "tab:blue", "tab:orange")
    ax.barh(
        importance_df["feature"],
        importance_df["mean_coef"],
        xerr=importance_df["std_coef"],
        color=colors,
        edgecolor="black",
        alpha=0.85,
        capsize=4,
    )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_title("Feature Importance from Logistic Regression")
    ax.set_xlabel("Mean raw coefficient across splits")
    ax.set_ylabel("Feature")

    x_pad = max(0.02, importance_df["mean_coef"].abs().max() * 0.04)
    for _, row in importance_df.iterrows():
        x = row["mean_coef"] + (x_pad if row["mean_coef"] >= 0 else -x_pad)
        ha = "left" if row["mean_coef"] >= 0 else "right"
        ax.text(x, row["feature"], f"{row['mean_coef']:+.3f}", va="center", ha=ha, fontsize=9)

    fig.text(
        0.5,
        0.01,
        f"Error bars show standard deviation across {N_SPLITS} train/test splits.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outdir / "feature_importance_logreg.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_compact_split_grid_pdf(
    page_specs: list[tuple[str, list[tuple[int, pd.DataFrame, dict]]]],
    outpath: Path,
) -> None:
    """Create a compact multipage PDF with vector 2x5 split plots."""
    nrows, ncols = 2, 5

    with PdfPages(outpath) as pdf:
        legend_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="None", color="w", label=TRUE_POSITIVE_LABEL,
                       markerfacecolor="tab:blue", markeredgecolor="black", markersize=6),
            plt.Line2D([0], [0], marker="o", linestyle="None", color="w", label=TRUE_NEGATIVE_LABEL,
                       markerfacecolor="tab:orange", markeredgecolor="black", markersize=6),
            plt.Line2D([0], [0], marker="o", linestyle="None", color="black", label=PRED_POSITIVE_LABEL, markersize=6),
            plt.Line2D([0], [0], marker="s", linestyle="None", color="black", label=PRED_NEGATIVE_LABEL, markersize=6),
        ]

        for page_title, split_records in page_specs:
            if len(split_records) != nrows * ncols:
                raise ValueError(
                    f"{page_title} needs exactly {nrows * ncols} plots, got {len(split_records)}."
                )

            fig, axes = plt.subplots(nrows, ncols, figsize=(18, 7.3), sharex=True, sharey=False)
            for ax, (split_id, split_df, split_metrics) in zip(axes.ravel(), split_records):
                colors = np.where(split_df[BINDER_COLUMN] == 1, "tab:blue", "tab:orange")
                markers = np.where(split_df[PRED_LABEL_COLUMN] == 1, "o", "s")

                for idx in range(split_df.shape[0]):
                    row = split_df.iloc[idx]
                    ax.scatter(
                        row[PRED_PROB_COLUMN],
                        row[TARGET_COLUMN],
                        color=colors[idx],
                        marker=markers[idx],
                        s=18,
                        alpha=0.9,
                        edgecolors="black",
                        linewidths=0.25,
                    )

                ax.axvline(PREDICTION_THRESHOLD, color="tab:red", linestyle="--", linewidth=0.8)
                ax.axhline(TARGET_THRESHOLD, color="black", linestyle=":", linewidth=0.8)
                add_point_labels(ax, split_df, compact=True)
                ax.set_title(
                    f"S{split_id:02d} a={split_metrics['accuracy']:.2f} u={split_metrics['roc_auc']:.2f}",
                    fontsize=6.5,
                    pad=2,
                )
                ax.tick_params(axis="both", labelsize=7, length=2)

            for ax in axes[-1, :]:
                ax.set_xlabel("p(binder)", fontsize=8)
            for ax in axes[:, 0]:
                ax.set_ylabel("Kd (uM)", fontsize=8)

            fig.suptitle(page_title, fontsize=10, y=0.988)
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=4,
                fontsize=7,
                frameon=False,
                bbox_to_anchor=(0.5, 0.012),
            )
            fig.subplots_adjust(left=0.05, right=0.995, top=0.905, bottom=0.10, wspace=0.18, hspace=0.33)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def build_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build constrained train/test splits."""
    binder_idx = df.index[df[BINDER_COLUMN] == 1].to_numpy()
    nonbinder_idx = df.index[df[BINDER_COLUMN] == 0].to_numpy()

    n_binder_test = TEST_BINDERS
    n_nonbinder_test = TEST_NONBINDERS

    if n_binder_test < 0:
        raise ValueError("Invalid split constraints: binder test count became negative.")
    if n_nonbinder_test > len(nonbinder_idx):
        raise ValueError("Invalid split constraints: too many non-binders requested in test set.")
    if n_binder_test > len(binder_idx):
        raise ValueError("Invalid split constraints: too many binders requested in test set.")

    rng = np.random.default_rng(RANDOM_SEED)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(N_SPLITS):
        test_nonbinder = rng.choice(nonbinder_idx, size=n_nonbinder_test, replace=False)
        test_binder = rng.choice(binder_idx, size=n_binder_test, replace=False)
        test_idx = np.concatenate([test_nonbinder, test_binder])
        train_idx = np.setdiff1d(df.index.to_numpy(), test_idx)
        splits.append((train_idx, test_idx))

    return splits



def save_split_csvs(df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    split_dir = OUTPUT_DIR / "datasets"
    split_dir.mkdir(parents=True, exist_ok=True)

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()
        train_df.to_csv(split_dir / f"split_{i:02d}_train.csv", index=False)
        test_df.to_csv(split_dir / f"split_{i:02d}_test.csv", index=False)



def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    split_plot_dir = OUTPUT_DIR / "split_test_plots"
    split_train_plot_dir = OUTPUT_DIR / "split_train_plots"
    split_plot_dir.mkdir(parents=True, exist_ok=True)
    split_train_plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    required_cols = [*FEATURE_NAMES, TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    model_df = df[[ID_COLUMN, *FEATURE_NAMES, TARGET_COLUMN]].copy()
    model_df = model_df.dropna(subset=[*FEATURE_NAMES, TARGET_COLUMN])
    model_df[BINDER_COLUMN] = (model_df[TARGET_COLUMN] < TARGET_THRESHOLD).astype(int)

    splits = build_splits(model_df)
    save_split_csvs(model_df, splits)

    X = model_df[FEATURE_NAMES].to_numpy()
    y = model_df[BINDER_COLUMN].to_numpy()

    rows = []
    raw_intercepts = []
    raw_coefs = []
    plt.figure(figsize=(8, 6))
    compact_test_records: list[tuple[int, pd.DataFrame, dict]] = []
    compact_train_records: list[tuple[int, pd.DataFrame, dict]] = []

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=MODEL_MAX_ITER, random_state=RANDOM_SEED + i)),
            ]
        )
        clf.fit(X_train, y_train)
        intercept_raw, coef_raw = get_logreg_raw_params(clf)
        raw_intercepts.append(intercept_raw)
        raw_coefs.append(coef_raw)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred_train = clf.predict(X_train)
        y_prob_train = clf.predict_proba(X_train)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        auc = np.nan
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, alpha=0.35, label=f"Split {i:02d} (AUC={auc:.2f})")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        rows.append(
            {
                "split": i,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "binders_in_test": int(y_test.sum()),
                "nonbinders_in_test": int((y_test == 0).sum()),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
        split_result = rows[-1]

        split_df = model_df.loc[test_idx, [ID_COLUMN, TARGET_COLUMN, BINDER_COLUMN]].copy()
        split_df[PRED_PROB_COLUMN] = y_prob
        split_df[PRED_LABEL_COLUMN] = y_pred
        split_df[CORRECT_COLUMN] = (split_df[BINDER_COLUMN] == split_df[PRED_LABEL_COLUMN]).astype(int)
        split_df.to_csv(OUTPUT_DIR / "datasets" / f"split_{i:02d}_test_predictions.csv", index=False)
        test_metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        }
        compact_test_records.append((i, split_df, test_metrics))
        make_split_separation_plot(
            i,
            split_df,
            split_result,
            split_plot_dir,
            dataset_name="Test",
            file_tag="test_separation",
        )

        train_acc = accuracy_score(y_train, y_pred_train)
        train_prec = precision_score(y_train, y_pred_train, zero_division=0)
        train_rec = recall_score(y_train, y_pred_train, zero_division=0)
        train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
        train_auc = np.nan
        if len(np.unique(y_train)) > 1:
            train_auc = roc_auc_score(y_train, y_prob_train)

        train_split_df = model_df.loc[train_idx, [ID_COLUMN, TARGET_COLUMN, BINDER_COLUMN]].copy()
        train_split_df[PRED_PROB_COLUMN] = y_prob_train
        train_split_df[PRED_LABEL_COLUMN] = y_pred_train
        train_split_df[CORRECT_COLUMN] = (train_split_df[BINDER_COLUMN] == train_split_df[PRED_LABEL_COLUMN]).astype(int)
        train_split_df.to_csv(OUTPUT_DIR / "datasets" / f"split_{i:02d}_train_predictions.csv", index=False)
        train_metrics = {
            "accuracy": train_acc,
            "precision": train_prec,
            "recall": train_rec,
            "f1": train_f1,
            "roc_auc": train_auc,
        }
        compact_train_records.append((i, train_split_df, train_metrics))
        make_split_separation_plot(
            i,
            train_split_df,
            train_metrics,
            split_train_plot_dir,
            dataset_name="Train",
            file_tag="train_separation",
        )

        print(f"\n=== Split {i:02d} ===")
        print(f"Train size: {len(train_idx)} | Test size: {len(test_idx)}")
        print(f"Test composition -> binders: {int(y_test.sum())}, non-binders: {int((y_test == 0).sum())}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"ROC-AUC: {auc:.3f}" if not np.isnan(auc) else "ROC-AUC: nan (single class in test)")
        print("Confusion matrix [ [TN FP] [FN TP] ]:")
        print(cm)
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=CLASSIFICATION_TARGET_NAMES, zero_division=0))
        print_logreg_formula(clf, i)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)

    mean_intercept = float(np.mean(raw_intercepts))
    mean_coef = np.mean(np.vstack(raw_coefs), axis=0)

    logits_all = mean_intercept + X @ mean_coef
    y_prob_all = 1.0 / (1.0 + np.exp(-logits_all))
    y_pred_all = (y_prob_all >= PREDICTION_THRESHOLD).astype(int)

    all_acc = accuracy_score(y, y_pred_all)
    all_prec = precision_score(y, y_pred_all, zero_division=0)
    all_rec = recall_score(y, y_pred_all, zero_division=0)
    all_f1 = f1_score(y, y_pred_all, zero_division=0)
    all_auc = np.nan
    if len(np.unique(y)) > 1:
        all_auc = roc_auc_score(y, y_prob_all)

    print("\n=== Averaged Model Across Splits (Applied to All Data) ===")
    coef_terms = " + ".join(
        f"({mean_coef[idx]:+.6f})*{feature}" for idx, feature in enumerate(FEATURE_NAMES)
    )
    print(f"p(binder) = sigmoid({mean_intercept:+.6f} + {coef_terms})")
    print(f"All-data Accuracy: {all_acc:.3f}")
    print(f"All-data Precision: {all_prec:.3f}")
    print(f"All-data Recall: {all_rec:.3f}")
    print(f"All-data F1-score: {all_f1:.3f}")
    print(f"All-data ROC-AUC: {all_auc:.3f}" if not np.isnan(all_auc) else "All-data ROC-AUC: nan")

    all_df = model_df[[ID_COLUMN, TARGET_COLUMN, BINDER_COLUMN]].copy()
    all_df[PRED_PROB_COLUMN] = y_prob_all
    all_df[PRED_LABEL_COLUMN] = y_pred_all
    all_df[CORRECT_COLUMN] = (all_df[BINDER_COLUMN] == all_df[PRED_LABEL_COLUMN]).astype(int)
    all_df.to_csv(OUTPUT_DIR / "all_data_predictions_avg_model.csv", index=False)
    all_metrics = {
        "accuracy": all_acc,
        "precision": all_prec,
        "recall": all_rec,
        "f1": all_f1,
        "roc_auc": all_auc,
    }
    make_global_separation_plot(all_df, all_metrics, OUTPUT_DIR)
    make_feature_importance_plot(raw_coefs, OUTPUT_DIR)

    compact_pdf_path = OUTPUT_DIR / "split_separation_compact.pdf"
    make_compact_split_grid_pdf(
        [
            ("Test split separation plots", compact_test_records),
            ("Train split separation plots", compact_train_records),
        ],
        compact_pdf_path,
    )

    print(f"\n=== Overall Summary ({N_SPLITS} datasets) ===")
    print(metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std", "min", "max"]).to_string())

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves Across {N_SPLITS} Test Splits")
    plt.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"roc_curves_{N_SPLITS}_splits.pdf", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["split"], metrics_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(metrics_df["split"], metrics_df["precision"], marker="o", label="Precision")
    plt.plot(metrics_df["split"], metrics_df["recall"], marker="o", label="Recall")
    plt.plot(metrics_df["split"], metrics_df["f1"], marker="o", label="F1")
    plt.xlabel("Split")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.title(f"Model Metrics Across {N_SPLITS} Datasets")
    plt.xticks(metrics_df["split"])
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_by_split.pdf", dpi=200)
    plt.close()

    print(f"\nSaved dataset splits, metrics, and plots in: {OUTPUT_DIR.resolve()}")
    print(f"Saved compact split PDF: {compact_pdf_path.resolve()}")


if __name__ == "__main__":
    run()
