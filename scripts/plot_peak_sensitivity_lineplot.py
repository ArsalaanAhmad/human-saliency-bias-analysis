from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Config
# ============================================================
SENSITIVITY_CSV = Path("outputs/all_models_peak_sensitivity_overall_mean.csv")
HUMAN_CSV = Path("outputs/human_per_image_behaviour.csv")
CENTER_BASELINE_CSV = Path("outputs/center_bias_baseline_overall_behaviour.csv")
TRANSALNET_OVERALL_CSV = Path("outputs/transalnet_overall_behaviour.csv")

# If your sensitivity csv has multiple neighborhoods, choose one for the x-axis line plot.
NEIGHBORHOOD_FOR_LINEPLOT = 25

OUTPUT_FIG = Path("outputs/figures/mean_peak_count_vs_threshold.png")

MODEL_ORDER = [
    "Human",
    "Centre Gaussian baseline",
    "DeepGaze IIE",
    "SAM-ResNet",
    "TranSalNet",
]

MODEL_COLORS = {
    "Human": "#111827",
    "Centre Gaussian baseline": "#6B7280",
    "DeepGaze IIE": "#2563EB",
    "SAM-ResNet": "#059669",
    "TranSalNet": "#D97706",
}


def load_constants() -> dict[str, float]:
    human_df = pd.read_csv(HUMAN_CSV)
    center_df = pd.read_csv(CENTER_BASELINE_CSV)
    trans_df = pd.read_csv(TRANSALNET_OVERALL_CSV)

    return {
        "Human": float(pd.to_numeric(human_df["human_num_peaks"], errors="coerce").mean()),
        "Centre Gaussian baseline": float(pd.to_numeric(center_df["model_num_peaks"], errors="coerce").iloc[0]),
        "TranSalNet": float(pd.to_numeric(trans_df["model_num_peaks"], errors="coerce").iloc[0]),
    }


def main() -> None:
    sens_df = pd.read_csv(SENSITIVITY_CSV)

    required = {"model_name", "threshold_ratio", "neighborhood", "mean_model_num_peaks"}
    missing = required - set(sens_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {SENSITIVITY_CSV}: {sorted(missing)}")

    sens_df = sens_df.copy()
    sens_df["threshold_ratio"] = pd.to_numeric(sens_df["threshold_ratio"], errors="coerce")
    sens_df["neighborhood"] = pd.to_numeric(sens_df["neighborhood"], errors="coerce")
    sens_df["mean_model_num_peaks"] = pd.to_numeric(sens_df["mean_model_num_peaks"], errors="coerce")
    sens_df = sens_df.dropna(subset=["threshold_ratio", "neighborhood", "mean_model_num_peaks"])

    line_df = sens_df[sens_df["neighborhood"] == NEIGHBORHOOD_FOR_LINEPLOT].copy()
    if line_df.empty:
        raise RuntimeError(
            f"No rows found for neighborhood={NEIGHBORHOOD_FOR_LINEPLOT} in {SENSITIVITY_CSV}."
        )

    # Map internal names to display names.
    line_df["model_display"] = line_df["model_name"].map(
        {
            "deepgazeiie": "DeepGaze IIE",
            "samresnet": "SAM-ResNet",
            "transalnet": "TranSalNet",
            "center_gaussian_baseline": "Centre Gaussian baseline",
        }
    )

    constants = load_constants()
    thresholds = sorted(line_df["threshold_ratio"].unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot model sensitivity lines from CSV (DeepGaze/SAM and TranSalNet if present).
    for model in ["DeepGaze IIE", "SAM-ResNet", "TranSalNet"]:
        model_slice = line_df[line_df["model_display"] == model].sort_values("threshold_ratio")
        if model_slice.empty:
            # Fallback to constant from overall CSV when threshold rows are missing.
            if model in constants:
                ax.plot(
                    thresholds,
                    [constants[model]] * len(thresholds),
                    label=f"{model} (overall constant)",
                    color=MODEL_COLORS[model],
                    linewidth=2.2,
                    linestyle="--",
                    marker="o",
                )
            continue

        ax.plot(
            model_slice["threshold_ratio"],
            model_slice["mean_model_num_peaks"],
            label=model,
            color=MODEL_COLORS[model],
            linewidth=2.6,
            marker="o",
        )

    # Plot constant references across thresholds.
    ax.plot(
        thresholds,
        [constants["Human"]] * len(thresholds),
        label="Human",
        color=MODEL_COLORS["Human"],
        linewidth=2.2,
        linestyle=":",
    )
    ax.plot(
        thresholds,
        [constants["Centre Gaussian baseline"]] * len(thresholds),
        label="Centre Gaussian baseline",
        color=MODEL_COLORS["Centre Gaussian baseline"],
        linewidth=2.2,
        linestyle="-.",
    )

    ax.set_title(
        f"Mean Peak Count vs Threshold (neighborhood={NEIGHBORHOOD_FOR_LINEPLOT})",
        fontsize=13,
    )
    ax.set_xlabel("Threshold ratio")
    ax.set_ylabel("Mean peak count")
    ax.set_xticks(thresholds)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2)

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=250)
    plt.close(fig)

    print(f"Saved: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
