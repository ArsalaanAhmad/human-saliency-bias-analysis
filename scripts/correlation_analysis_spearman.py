from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIG (edit these values)
# ============================================================
INPUT_CSV = Path("outputs/all_models_human_delta_per_image.csv")

OUTPUT_OVERALL_CSV = Path("outputs/spearman_correlations_overall.csv")
OUTPUT_PER_CATEGORY_CSV = Path("outputs/spearman_correlations_per_category.csv")

HEATMAP_DIR = Path("outputs/figures/correlations")
SCATTER_DIR = Path("outputs/figures/correlations/scatterplots")

# Correlation metric columns.
BENCHMARK_METRICS = ["model_nss", "model_cc", "model_auc"]
STRUCTURAL_METRICS = ["delta_entropy", "delta_num_peaks", "delta_center_distance"]

# Optional analyses.
COMPUTE_PER_CATEGORY = True
GENERATE_SCATTERPLOTS = True

# Scatterplot pairs: (x_metric, y_metric).
# These are the pairs requested in your prompt; add more if needed.
SCATTER_PAIRS: List[Tuple[str, str]] = [
    ("delta_num_peaks", "model_nss"),
    ("delta_center_distance", "model_cc"),
]

# Plot style.
DPI = 260
HEATMAP_FIGSIZE = (8.2, 4.8)
SCATTER_FIGSIZE = (6.5, 5.2)
ANNOTATE_CELLS = True
ANNOTATION_FONTSIZE = 10


# ============================================================
# Helpers
# ============================================================
def ensure_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """
    Compute Spearman correlation safely with pairwise NA dropping.

    Returns:
    - rho: correlation coefficient
    - p_value: significance p-value
    - n: number of valid paired samples
    """
    pair = pd.concat([x, y], axis=1).dropna()
    n = int(len(pair))

    if n < 3:
        return np.nan, np.nan, n

    rho, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(rho), float(p), n


def compute_overall_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    for model_name, model_df in df.groupby("model_name", dropna=True):
        for benchmark_metric in BENCHMARK_METRICS:
            for structural_metric in STRUCTURAL_METRICS:
                rho, p_value, n = safe_spearman(model_df[benchmark_metric], model_df[structural_metric])
                rows.append(
                    {
                        "model_name": str(model_name),
                        "benchmark_metric": benchmark_metric,
                        "structural_metric": structural_metric,
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "n_pairs": n,
                    }
                )

    return pd.DataFrame(rows)


def compute_per_category_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    grouped = df.groupby(["model_name", "category"], dropna=True)
    for (model_name, category), chunk in grouped:
        for benchmark_metric in BENCHMARK_METRICS:
            for structural_metric in STRUCTURAL_METRICS:
                rho, p_value, n = safe_spearman(chunk[benchmark_metric], chunk[structural_metric])
                rows.append(
                    {
                        "model_name": str(model_name),
                        "category": str(category),
                        "benchmark_metric": benchmark_metric,
                        "structural_metric": structural_metric,
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "n_pairs": n,
                    }
                )

    return pd.DataFrame(rows)


def format_model_name_for_filename(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def pretty_metric_name(metric: str) -> str:
    labels = {
        "model_nss": "NSS",
        "model_cc": "CC",
        "model_auc": "AUC-Judd",
        "delta_entropy": "Delta Entropy",
        "delta_num_peaks": "Delta Peak Count",
        "delta_center_distance": "Delta Center Distance",
    }
    return labels.get(metric, metric)


def build_heatmap_matrix(model_corr_df: pd.DataFrame) -> np.ndarray:
    """Build a 3x3 matrix: rows benchmark metrics, columns structural metrics."""
    mat = np.full((len(BENCHMARK_METRICS), len(STRUCTURAL_METRICS)), np.nan, dtype=float)

    for i, bench in enumerate(BENCHMARK_METRICS):
        for j, struct in enumerate(STRUCTURAL_METRICS):
            hit = model_corr_df[
                (model_corr_df["benchmark_metric"] == bench)
                & (model_corr_df["structural_metric"] == struct)
            ]
            if not hit.empty:
                mat[i, j] = float(hit.iloc[0]["spearman_rho"])

    return mat


def plot_model_heatmap(model_name: str, matrix: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)

    # Diverging colormap centered around zero.
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(STRUCTURAL_METRICS)))
    ax.set_yticks(np.arange(len(BENCHMARK_METRICS)))
    ax.set_xticklabels([pretty_metric_name(m) for m in STRUCTURAL_METRICS], rotation=22, ha="right")
    ax.set_yticklabels([pretty_metric_name(m) for m in BENCHMARK_METRICS])

    ax.set_xlabel("Structural Mismatch Metrics")
    ax.set_ylabel("Benchmark Metrics")
    ax.set_title(f"{model_name}: Spearman Correlation Matrix")

    # Light cell borders to improve readability for dissertation figures.
    ax.set_xticks(np.arange(-0.5, len(STRUCTURAL_METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BENCHMARK_METRICS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if ANNOTATE_CELLS:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text = "nan" if np.isnan(val) else f"{val:.2f}"
                # Choose text color for contrast.
                color = "white" if (not np.isnan(val) and abs(val) > 0.5) else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=ANNOTATION_FONTSIZE)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Spearman rho")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_scatterpairs_for_model(df_model: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = format_model_name_for_filename(model_name)

    for x_metric, y_metric in SCATTER_PAIRS:
        # Drop missing values pairwise.
        pair = df_model[[x_metric, y_metric]].dropna()
        if pair.empty:
            continue

        rho, p_value, n = safe_spearman(pair[x_metric], pair[y_metric])

        fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
        ax.scatter(pair[x_metric], pair[y_metric], s=12, alpha=0.55, color="#1f77b4")

        ax.set_xlabel(pretty_metric_name(x_metric))
        ax.set_ylabel(pretty_metric_name(y_metric))
        ax.set_title(f"{model_name}: {pretty_metric_name(y_metric)} vs {pretty_metric_name(x_metric)}")
        ax.grid(alpha=0.25, linestyle="--")

        subtitle = f"Spearman rho={rho:.3f} | p={p_value:.3g} | n={n}" if not np.isnan(rho) else f"Spearman rho=nan | n={n}"
        ax.text(
            0.02,
            0.98,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        out_file = out_dir / f"{safe_model}_{x_metric}_vs_{y_metric}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=DPI)
        plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "model_name",
        "category",
        "image",
        *BENCHMARK_METRICS,
        *STRUCTURAL_METRICS,
    ]
    ensure_columns(df, required_cols)

    # Convert metric columns to numeric safely.
    for col in BENCHMARK_METRICS + STRUCTURAL_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Overall correlations per model.
    overall_corr = compute_overall_correlations(df)
    OUTPUT_OVERALL_CSV.parent.mkdir(parents=True, exist_ok=True)
    overall_corr.to_csv(OUTPUT_OVERALL_CSV, index=False)
    print(f"Saved: {OUTPUT_OVERALL_CSV}")

    # Optional: per-category correlations.
    if COMPUTE_PER_CATEGORY:
        per_cat_corr = compute_per_category_correlations(df)
        OUTPUT_PER_CATEGORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        per_cat_corr.to_csv(OUTPUT_PER_CATEGORY_CSV, index=False)
        print(f"Saved: {OUTPUT_PER_CATEGORY_CSV}")

    # Heatmap per model.
    for model_name in sorted(df["model_name"].dropna().astype(str).unique().tolist()):
        model_corr = overall_corr[overall_corr["model_name"] == model_name].copy()
        matrix = build_heatmap_matrix(model_corr)

        safe_model = format_model_name_for_filename(model_name)
        out_file = HEATMAP_DIR / f"{safe_model}_correlation_heatmap.png"
        plot_model_heatmap(model_name, matrix, out_file)
        print(f"Saved: {out_file}")

    # Optional scatterplots per model.
    if GENERATE_SCATTERPLOTS:
        for model_name, model_df in df.groupby("model_name", dropna=True):
            plot_scatterpairs_for_model(model_df, str(model_name), SCATTER_DIR)
        print(f"Saved scatterplots under: {SCATTER_DIR}")


if __name__ == "__main__":
    main()
