from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Easy-to-edit paths
INPUT_CSV = Path("outputs/all_models_human_delta_per_image.csv")
OUTPUT_CSV = Path("outputs/model_metric_confidence_intervals.csv")

# Metrics to summarize per model
METRICS = [
    "model_nss",
    "model_cc",
    "model_auc",
    "delta_entropy",
    "delta_center_distance",
    "delta_num_peaks",
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

    # Draw bootstrap samples of shape (iterations, n) and compute means.
    sample_indices = rng.integers(0, n, size=(iterations, n))
    sample_means = values[sample_indices].mean(axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return lower, upper


def normal_approx_ci_mean(mean: float, std: float, n: int, confidence_level: float) -> tuple[float, float]:
    """Fallback normal-approximation CI for the mean."""
    if n <= 1 or np.isnan(std):
        return np.nan, np.nan

    # For 95% CI, z is approximately 1.96. For other levels, this simple fallback
    # still uses 1.96 to keep dependencies minimal.
    z = 1.96
    margin = z * (std / np.sqrt(n))
    return mean - margin, mean + margin


def metric_summary_for_model(df_model: pd.DataFrame, model_name: str) -> list[dict]:
    rows: list[dict] = []

    for metric in METRICS:
        if metric not in df_model.columns:
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
            continue

        values = pd.to_numeric(df_model[metric], errors="coerce").dropna().to_numpy(dtype=float)
        n = int(values.size)

        if n == 0:
            rows.append(
                {
                    "model_name": model_name,
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
            ci_lower, ci_upper = bootstrap_ci_mean(
                values=values,
                iterations=BOOTSTRAP_ITERATIONS,
                confidence_level=CONFIDENCE_LEVEL,
                seed=RANDOM_SEED,
            )
        else:
            ci_lower, ci_upper = normal_approx_ci_mean(
                mean=mean,
                std=std,
                n=n,
                confidence_level=CONFIDENCE_LEVEL,
            )

        rows.append(
            {
                "model_name": model_name,
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

    if "model_name" not in df.columns:
        raise ValueError("Input CSV must contain a 'model_name' column.")

    all_rows: list[dict] = []

    for model_name, df_model in df.groupby("model_name", dropna=False):
        safe_model_name = str(model_name) if pd.notna(model_name) else "<missing_model_name>"
        all_rows.extend(metric_summary_for_model(df_model=df_model, model_name=safe_model_name))

    summary_df = pd.DataFrame(all_rows)
    summary_df = summary_df.sort_values(["model_name", "metric"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved confidence intervals to: {OUTPUT_CSV}")
    print()

    pd.options.display.float_format = "{:.6f}".format
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
