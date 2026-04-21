from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG (edit this section)
# ============================================================
MODEL_FILES: Dict[str, Path] = {
    "Centre-Bias Baseline": Path("outputs/center_bias_baseline_human_delta_per_category_mean.csv"),
    "DeepGaze IIE": Path("outputs/deepgazeiie_human_delta_per_category_mean.csv"),
    "SAM-ResNet": Path("outputs/samresnet_human_delta_per_category_mean.csv"),
    "TranSalNet": Path("outputs/transalnet_human_delta_per_category_mean.csv"),
}

# Explicit order for bars and legend across all figures.
MODEL_ORDER: List[str] = [
    "DeepGaze IIE",
    "SAM-ResNet",
    "TranSalNet",
    "Centre-Bias Baseline",
]

# Keep these colors fixed so model identity stays consistent across all figures.
MODEL_COLORS: Dict[str, str] = {
    "Centre-Bias Baseline": "#6B7280",
    "DeepGaze IIE": "#2563EB",
    "SAM-ResNet": "#059669",
    "TranSalNet": "#D97706",
}

# metric_column, figure_title, output_path
FIGURE_SPECS: List[Tuple[str, str, Path]] = [
    (
        "delta_entropy",
        "Figure 4.3a - Delta Entropy by Category",
        Path("outputs/figures/Figure 4.3a - Delta Entropy by Category.png"),
    ),
    (
        "delta_num_peaks",
        "Figure 4.3b - Delta Peak Count by Category",
        Path("outputs/figures/Figure 4.3b - Delta Peak Count by Category.png"),
    ),
    (
        "delta_center_distance",
        "Figure 4.3c - Delta Centre Distance by Category",
        Path("outputs/figures/Figure 4.3c - Delta Centre Distance by Category.png"),
    ),
]

OUTPUT_MAIN_FIGURE = Path("outputs/figures/Figure 4.3 - Structural Behavioural Comparison.png")

# Sorting and style controls.
SORT_CATEGORIES_BY_MEAN = True
FIG_WIDTH = 13
BAR_HEIGHT_MAX = 0.18
ROW_HEIGHT = 0.50
DPI = 320


# ============================================================
# Data loading helpers
# ============================================================
def load_one_model(
    model_name: str,
    csv_path: Path,
    required_metrics: List[str],
) -> pd.DataFrame | None:
    """Load one model CSV safely and return harmonized rows.

    Returns None if file is missing/invalid so plotting can continue with available models.
    """
    if not csv_path.exists():
        print(f"[WARN] Missing file for {model_name}: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    required = {"category", *required_metrics}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Skipping {model_name}; missing columns {sorted(missing)} in {csv_path}")
        return None

    out = df[["category", *required_metrics]].copy()
    out["model_name"] = model_name

    out["category"] = out["category"].astype(str)
    for metric in required_metrics:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")

    out = out.dropna(subset=["category"]).reset_index(drop=True)
    return out


def load_all_models(
    model_files: Dict[str, Path],
    figure_specs: List[Tuple[str, str, Path]],
) -> pd.DataFrame:
    """Load all available model CSVs and concatenate into one dataframe."""
    metric_cols = [metric for metric, _, _ in figure_specs]
    parts: List[pd.DataFrame] = []

    for model_name, csv_path in model_files.items():
        one = load_one_model(model_name=model_name, csv_path=csv_path, required_metrics=metric_cols)
        if one is not None:
            parts.append(one)

    if not parts:
        raise RuntimeError("No valid model CSV files were loaded. Check MODEL_FILES and CSV columns.")

    return pd.concat(parts, ignore_index=True)


# ============================================================
# Plot helpers
# ============================================================
def category_order(df: pd.DataFrame, metric_cols: List[str], sort_by_mean: bool) -> List[str]:
    """Build a stable category order for all 3 figures."""
    categories = sorted(df["category"].dropna().astype(str).unique().tolist())
    if not sort_by_mean:
        return categories

    category_scores = (
        df.groupby("category", as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )
    category_scores["mean_score"] = category_scores[metric_cols].mean(axis=1)

    ordered = (
        category_scores.sort_values("mean_score", ascending=False)["category"]
        .astype(str)
        .tolist()
    )
    remainder = [cat for cat in categories if cat not in ordered]
    return ordered + remainder


def build_metric_table(
    df: pd.DataFrame,
    categories: List[str],
    models: List[str],
    metric_col: str,
) -> pd.DataFrame:
    """Create category x model table for one metric."""
    table = (
        df.pivot_table(index="category", columns="model_name", values=metric_col, aggfunc="mean")
        .reindex(index=categories, columns=models)
    )
    return table


def plot_grouped_horizontal_bars(
    metric_table: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
    model_colors: Dict[str, str],
) -> None:
    """Plot one horizontal grouped bar chart with one bar per model per category."""
    categories = metric_table.index.astype(str).tolist()
    models = metric_table.columns.astype(str).tolist()

    n_categories = len(categories)
    n_models = max(1, len(models))

    fig_height = max(8.5, n_categories * ROW_HEIGHT)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    y_base = np.arange(n_categories)
    bar_height = min(0.84 / n_models, BAR_HEIGHT_MAX)
    offsets = np.linspace(
        -(n_models - 1) * bar_height / 2,
        +(n_models - 1) * bar_height / 2,
        n_models,
    )

    default_colors = plt.get_cmap("tab10", n_models)

    # Draw grouped bars category-by-category.
    for idx, (model_name, offset) in enumerate(zip(models, offsets)):
        vals = pd.to_numeric(metric_table[model_name], errors="coerce").to_numpy(dtype=float)
        color = model_colors.get(model_name, default_colors(idx))
        ax.barh(
            y_base + offset,
            vals,
            height=bar_height,
            color=color,
            alpha=0.92,
            label=model_name,
        )

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(metric_col, fontsize=11)
    ax.set_ylabel("Category", fontsize=11)

    ax.set_yticks(y_base)
    ax.set_yticklabels(categories, fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Highest-ranked category appears at top.
    ax.invert_yaxis()

    # Move legend to the bottom so the key index is below the chart.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=max(2, n_models),
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.10, 0.07, 0.98, 0.97])
    fig.savefig(output_path, dpi=DPI)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_main_figure_with_subfigures(
    metric_tables: Dict[str, pd.DataFrame],
    figure_specs: List[Tuple[str, str, Path]],
    output_path: Path,
    model_colors: Dict[str, str],
) -> None:
    """Create one main figure that contains the 3 metric subfigures (a, b, c)."""
    if not figure_specs:
        return

    first_metric = figure_specs[0][0]
    first_table = metric_tables[first_metric]
    n_categories = len(first_table.index)
    n_models = max(1, len(first_table.columns))

    # Large enough for 20 categories while still fitting thesis pages when scaled.
    fig_height = max(16.0, n_categories * 0.27 * len(figure_specs))
    fig, axes = plt.subplots(nrows=len(figure_specs), ncols=1, figsize=(FIG_WIDTH, fig_height), sharey=False)

    if len(figure_specs) == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None

    for ax, (metric_col, title, _) in zip(axes, figure_specs):
        metric_table = metric_tables[metric_col]
        categories = metric_table.index.astype(str).tolist()
        models = metric_table.columns.astype(str).tolist()

        y_base = np.arange(len(categories))
        bar_height = min(0.84 / n_models, BAR_HEIGHT_MAX)
        offsets = np.linspace(
            -(n_models - 1) * bar_height / 2,
            +(n_models - 1) * bar_height / 2,
            n_models,
        )

        default_colors = plt.get_cmap("tab10", n_models)

        for idx, (model_name, offset) in enumerate(zip(models, offsets)):
            vals = pd.to_numeric(metric_table[model_name], errors="coerce").to_numpy(dtype=float)
            color = model_colors.get(model_name, default_colors(idx))
            ax.barh(
                y_base + offset,
                vals,
                height=bar_height,
                color=color,
                alpha=0.92,
                label=model_name,
            )

        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel(metric_col, fontsize=10)
        ax.set_ylabel("Category", fontsize=10)
        ax.set_yticks(y_base)
        ax.set_yticklabels(categories, fontsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.invert_yaxis()

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.003),
            ncol=max(2, n_models),
            frameon=False,
        )

    fig.suptitle("Figure 4.3 - Structural Behavioural Comparison", fontsize=16, y=0.995)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.09, 0.04, 0.98, 0.98])
    fig.savefig(output_path, dpi=DPI)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    merged = load_all_models(model_files=MODEL_FILES, figure_specs=FIGURE_SPECS)

    metric_cols = [metric for metric, _, _ in FIGURE_SPECS]
    categories = category_order(
        df=merged,
        metric_cols=metric_cols,
        sort_by_mean=SORT_CATEGORIES_BY_MEAN,
    )

    # Preserve config ordering for model columns where possible.
    present_models = merged["model_name"].astype(str).unique().tolist()
    models = [model for model in MODEL_ORDER if model in present_models]
    models += [model for model in present_models if model not in models]

    metric_tables: Dict[str, pd.DataFrame] = {}
    for metric_col, title, output_path in FIGURE_SPECS:
        table = build_metric_table(
            df=merged,
            categories=categories,
            models=models,
            metric_col=metric_col,
        )
        metric_tables[metric_col] = table
        plot_grouped_horizontal_bars(
            metric_table=table,
            metric_col=metric_col,
            title=title,
            output_path=output_path,
            model_colors=MODEL_COLORS,
        )
        print(f"Saved: {output_path}")
        print(f"Saved: {output_path.with_suffix('.pdf')}")

    plot_main_figure_with_subfigures(
        metric_tables=metric_tables,
        figure_specs=FIGURE_SPECS,
        output_path=OUTPUT_MAIN_FIGURE,
        model_colors=MODEL_COLORS,
    )
    print(f"Saved: {OUTPUT_MAIN_FIGURE}")
    print(f"Saved: {OUTPUT_MAIN_FIGURE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
