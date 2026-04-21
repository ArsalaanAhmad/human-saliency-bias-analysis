from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("outputs/human_metric_confidence_intervals_by_category.csv")
OUTPUT_PNG = Path("outputs/human_metrics_by_category_3panel_ci.png")

# Metric display config: (metric_key, panel_title, x_label, color)
METRIC_CONFIG = [
    ("human_entropy", "Human Entropy by Category", "Mean entropy", "#3B82F6"),
    ("human_num_peaks", "Human Peak Count by Category", "Mean peak count", "#10B981"),
    ("human_center_distance", "Human Centre Distance by Category", "Mean centre distance", "#F59E0B"),
]


def metric_table(df: pd.DataFrame, metric_key: str, category_order: list[str]) -> pd.DataFrame:
    """Extract one metric and align rows to a common category order."""
    sub = df[df["metric"] == metric_key].copy()
    sub = sub.set_index("category").reindex(category_order).reset_index()
    return sub


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"category", "metric", "mean", "ci_lower", "ci_upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

    # Keep a stable, readable category order.
    category_order = sorted(df["category"].dropna().astype(str).unique().tolist())
    y_positions = np.arange(len(category_order))

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True)

    for ax, (metric_key, title, x_label, color) in zip(axes, METRIC_CONFIG):
        sub = metric_table(df=df, metric_key=metric_key, category_order=category_order)

        means = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
        ci_low = pd.to_numeric(sub["ci_lower"], errors="coerce").to_numpy(dtype=float)
        ci_high = pd.to_numeric(sub["ci_upper"], errors="coerce").to_numpy(dtype=float)

        # Asymmetric horizontal error bars from percentile CI bounds.
        xerr = np.vstack([means - ci_low, ci_high - means])

        ax.barh(y_positions, means, color=color, alpha=0.85)
        ax.errorbar(
            means,
            y_positions,
            xerr=xerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
        )

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel(x_label)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(category_order)
    axes[0].set_ylabel("Category")

    # Highest category appears at top for easier scanning.
    for ax in axes:
        ax.invert_yaxis()

    fig.suptitle("Human Category Metrics with 95% Confidence Intervals", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
