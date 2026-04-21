from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Easy-to-edit paths
INPUT_CSV = Path("outputs/human_per_image_behaviour.csv")
OUTPUT_OVERALL_CSV = Path("outputs/human_metric_confidence_intervals_overall.csv")
OUTPUT_BY_CATEGORY_CSV = Path("outputs/human_metric_confidence_intervals_by_category.csv")

# Human metrics to summarize
METRICS = [
    "human_entropy",
    "human_center_distance",
    "human_num_peaks",
]

# CI configuration
USE_BOOTSTRAP = True
BOOTSTRAP_ITERATIONS = 2000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def bootstrap_ci_mean(values: np.ndarray, iterations: int, confidence_level: float, seed: int) -> tuple[float, float]:
    """Compute bootstrap percentile CI for the mean using a fixed RNG seed."""
    rng = np.random.default_rng(seed)
    n = values.size

    sample_indices = rng.integers(0, n, size=(iterations, n))
    sample_means = values[sample_indices].mean(axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return lower, upper


def normal_approx_ci_mean(mean: float, std: float, n: int) -> tuple[float, float]:
    """Fallback normal-approximation CI for the mean."""
    if n <= 1 or np.isnan(std):
        return np.nan, np.nan

    z = 1.96
    margin = z * (std / np.sqrt(n))
    return mean - margin, mean + margin


def metric_summary(df_slice: pd.DataFrame, group_value: str, group_col_name: str) -> list[dict]:
    rows: list[dict] = []

    for metric in METRICS:
        if metric not in df_slice.columns:
            rows.append(
                {
                    group_col_name: group_value,
                    "metric": metric,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
            continue

        values = pd.to_numeric(df_slice[metric], errors="coerce").dropna().to_numpy(dtype=float)
        n = int(values.size)

        if n == 0:
            rows.append(
                {
                    group_col_name: group_value,
                    "metric": metric,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
            continue

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else np.nan

        if n == 1:
            ci_lower, ci_upper = mean, mean
        elif USE_BOOTSTRAP:
            # Use a deterministic per-group/per-metric seed so results are stable.
            seed_offset = abs(hash((group_value, metric))) % 100000
            ci_lower, ci_upper = bootstrap_ci_mean(
                values=values,
                iterations=BOOTSTRAP_ITERATIONS,
                confidence_level=CONFIDENCE_LEVEL,
                seed=RANDOM_SEED + seed_offset,
            )
        else:
            ci_lower, ci_upper = normal_approx_ci_mean(mean=mean, std=std, n=n)

        rows.append(
            {
                group_col_name: group_value,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    return rows


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if "category" not in df.columns:
        raise ValueError("Input CSV must contain a 'category' column.")

    overall_rows = metric_summary(df_slice=df, group_value="human", group_col_name="group_name")
    overall_df = pd.DataFrame(overall_rows).sort_values(["group_name", "metric"]).reset_index(drop=True)

    category_rows: list[dict] = []
    for category, df_cat in df.groupby("category", dropna=False):
        safe_category = str(category) if pd.notna(category) else "<missing_category>"
        category_rows.extend(metric_summary(df_slice=df_cat, group_value=safe_category, group_col_name="category"))

    by_category_df = pd.DataFrame(category_rows).sort_values(["category", "metric"]).reset_index(drop=True)

    OUTPUT_OVERALL_CSV.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(OUTPUT_OVERALL_CSV, index=False)
    by_category_df.to_csv(OUTPUT_BY_CATEGORY_CSV, index=False)

    print(f"Saved: {OUTPUT_OVERALL_CSV}")
    print(f"Saved: {OUTPUT_BY_CATEGORY_CSV}")


if __name__ == "__main__":
    main()
