from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import maximum_filter
from scipy.stats import entropy
from tqdm import tqdm

from metrics import auc_judd, cc, density_from_fixation_map, nss


# CAT2000 fixation-map root (contains category folders with .mat files).
FIX_ROOT = Path("data/FIXATIONLOCS")
OUTPUT_DIR = Path("outputs")
MODEL_NAME = "center_bias_baseline"

# Build human fixation density for CC using Gaussian smoothing.
GAUSSIAN_DENSITY_SIGMA = 15


def compute_entropy(prob_map: np.ndarray) -> float:
    """Compute Shannon entropy on a sum-normalized probability map."""
    p = prob_map.astype(np.float32).ravel()
    p = p / (p.sum() + 1e-8)
    return float(entropy(p + 1e-8))


def center_distance(prob_map: np.ndarray) -> float:
    """Probability-weighted average Euclidean distance to image center."""
    h, w = prob_map.shape
    cy, cx = h / 2.0, w / 2.0

    ys, xs = np.indices((h, w))
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    p = prob_map / (prob_map.sum() + 1e-8)
    return float((p * dists).sum())


def count_peaks(prob_map: np.ndarray, threshold_ratio: float = 0.6, neighborhood: int = 25) -> int:
    """Count local maxima above threshold_ratio * max_value."""
    max_value = float(prob_map.max())
    if max_value <= 0:
        return 0

    norm = prob_map / max_value
    local_max = norm == maximum_filter(norm, size=neighborhood)
    peaks = local_max & (norm >= threshold_ratio)
    return int(peaks.sum())


def generate_centered_gaussian_map(height: int, width: int, sigma_ratio: float = 0.2) -> np.ndarray:
    """
    Generate a centered 2D Gaussian map and sum-normalize it to probability mass 1.

    sigma = sigma_ratio * min(height, width)
    """
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2.0, width / 2.0
    sigma = float(sigma_ratio * min(height, width))

    # Guard against very small images or invalid ratio.
    sigma = max(sigma, 1e-6)

    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    total = float(gaussian.sum())
    if total <= 0:
        return np.zeros((height, width), dtype=np.float32)
    return (gaussian / total).astype(np.float32)


def load_fixation_map(fixation_path: Path) -> np.ndarray | None:
    """Load fixation map from .mat key 'fixLocs'; return None when invalid."""
    try:
        mat = loadmat(fixation_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {fixation_path}: {exc}")
        return None

    if "fixLocs" not in mat:
        print(f"[WARN] Missing 'fixLocs' in {fixation_path}")
        return None

    fixation_map = np.asarray(mat["fixLocs"], dtype=np.float32)
    if fixation_map.ndim != 2 or fixation_map.size == 0:
        print(f"[WARN] Invalid fixation map shape {fixation_map.shape} in {fixation_path}")
        return None

    return fixation_map


def run_evaluation(
    fix_root: Path,
    output_dir: Path,
    sigma_ratio: float,
    threshold_ratio: float,
    neighborhood: int,
    max_images_per_category: int | None,
) -> None:
    rows: list[dict[str, object]] = []

    categories = sorted([p for p in fix_root.iterdir() if p.is_dir()])
    if not categories:
        raise RuntimeError(f"No category folders found in {fix_root}")

    for category_dir in categories:
        mat_files = sorted(category_dir.glob("*.mat"))
        if max_images_per_category is not None and max_images_per_category > 0:
            mat_files = mat_files[:max_images_per_category]

        for fixation_path in tqdm(mat_files, desc=f"Analysing {category_dir.name}"):

            fixation_map = load_fixation_map(fixation_path)
            if fixation_map is None:
                continue

            h, w = fixation_map.shape

            # Center-bias baseline prediction with spread proportional to image size.
            prob_map = generate_centered_gaussian_map(h, w, sigma_ratio=sigma_ratio)

            # Human fixation density target for CC: Gaussian smoothing + max normalization.
            fixation_density = density_from_fixation_map(fixation_map, sigma=GAUSSIAN_DENSITY_SIGMA)

            rows.append(
                {
                    "category": category_dir.name,
                    "image": fixation_path.stem,
                    "model_entropy": compute_entropy(prob_map),
                    "model_center_distance": center_distance(prob_map),
                    "model_num_peaks": count_peaks(
                        prob_map,
                        threshold_ratio=threshold_ratio,
                        neighborhood=neighborhood,
                    ),
                    "model_nss": nss(prob_map, fixation_map),
                    "model_cc": cc(prob_map, fixation_density),
                    "model_auc": auc_judd(prob_map, fixation_map),
                }
            )

    if not rows:
        raise RuntimeError("No valid fixation maps were processed.")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "model_entropy",
        "model_center_distance",
        "model_num_peaks",
        "model_nss",
        "model_cc",
        "model_auc",
    ]

    # Per-image output schema matches learned-model files exactly.
    per_image_columns = ["category", "image", *metric_columns]
    per_image_df = df[per_image_columns]
    per_image_output = output_dir / f"{MODEL_NAME}_per_image_behaviour.csv"
    per_image_df.to_csv(per_image_output, index=False)

    # Per-category output keeps the same metric column names.
    per_category_df = per_image_df.groupby("category", as_index=False)[metric_columns].mean()
    per_category_output = output_dir / f"{MODEL_NAME}_per_category_behaviour.csv"
    per_category_df.to_csv(per_category_output, index=False)

    # Overall summary also uses the same metric column names.
    overall_summary = per_image_df[metric_columns].mean().to_dict()
    overall_summary.update(
        {
            "model": MODEL_NAME,
            "num_images": int(len(per_image_df)),
            "num_categories": int(per_image_df["category"].nunique()),
        }
    )
    overall_columns = ["model", "num_images", "num_categories", *metric_columns]
    overall_df = pd.DataFrame([overall_summary], columns=overall_columns)
    overall_output = output_dir / f"{MODEL_NAME}_overall_behaviour.csv"
    overall_df.to_csv(overall_output, index=False)

    print("Saved:")
    print(f" - {per_image_output}")
    print(f" - {per_category_output}")
    print(f" - {overall_output}")
    print(f"Processed images: {len(per_image_df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a centered Gaussian center-bias baseline on CAT2000 fixation maps."
    )
    parser.add_argument(
        "--fix-root",
        type=Path,
        default=FIX_ROOT,
        help="Path to CAT2000 fixation root (category folders containing .mat fixation maps).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where output CSV files will be written.",
    )
    parser.add_argument(
        "--sigma-ratio",
        type=float,
        default=0.2,
        help="Gaussian spread ratio: sigma = sigma_ratio * min(height, width).",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.6,
        help="Peak threshold as ratio of map maximum.",
    )
    parser.add_argument(
        "--neighborhood",
        type=int,
        default=25,
        help="Neighborhood size for local-max peak counting.",
    )
    parser.add_argument(
        "--max-images-per-category",
        type=int,
        default=50,
        help="Optional cap per category (default: 50). Use 0 or negative to disable cap.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        fix_root=args.fix_root,
        output_dir=args.output_dir,
        sigma_ratio=args.sigma_ratio,
        threshold_ratio=args.threshold_ratio,
        neighborhood=args.neighborhood,
        max_images_per_category=args.max_images_per_category,
    )









