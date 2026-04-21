from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import maximum_filter
from scipy.stats import entropy
from tqdm import tqdm

from metrics import auc_judd, cc, density_from_fixation_map, nss


FIX_ROOT = Path("data/FIXATIONLOCS")
PRED_ROOT = Path("D:/outputs/transalnet_cat2000_50")
MAX_IMAGES_PER_CATEGORY = int(os.getenv("MAX_IMAGES_PER_CATEGORY", "50"))
GAUSSIAN_SIGMA = 15

OUT_OVERALL = Path("outputs/tables/summary/transalnet_preprocessing_sweep_overall.csv")
OUT_PER_CATEGORY = Path("outputs/tables/summary/transalnet_preprocessing_sweep_per_category.csv")

PREDICTION_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.npy"]


def compute_entropy(prob_map: np.ndarray) -> float:
    p = prob_map.flatten().astype(np.float32)
    p = p / (p.sum() + 1e-8)
    return float(entropy(p + 1e-8))


def center_distance(prob_map: np.ndarray) -> float:
    h, w = prob_map.shape
    cy, cx = h / 2.0, w / 2.0
    ys, xs = np.indices((h, w))
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    p = prob_map / (prob_map.sum() + 1e-8)
    return float((p * dists).sum())


def count_peaks(prob_map: np.ndarray, threshold_ratio: float = 0.6, neighborhood: int = 25) -> int:
    max_value = float(prob_map.max())
    if max_value <= 0:
        return 0
    norm = prob_map / max_value
    local_max = norm == maximum_filter(norm, size=neighborhood)
    peaks = local_max & (norm >= threshold_ratio)
    return int(peaks.sum())


def load_prediction_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def to_prob_map(pred: np.ndarray, mode: str) -> np.ndarray:
    x = pred.astype(np.float32)

    if mode == "plain":
        x = x - float(x.min())
    elif mode == "inverted":
        x = float(x.max()) - x
    elif mode == "log_density_like":
        x = np.exp(x - float(x.max()))
    elif mode == "inverted_log_density_like":
        x = float(x.max()) - x
        x = np.exp(x - float(x.max()))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(x.sum())
    if total <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return (x / total).astype(np.float32)


def load_fixation_map(category: str, image_stem: str, target_shape: tuple[int, int]) -> np.ndarray | None:
    path = FIX_ROOT / category / f"{image_stem}.mat"
    if not path.exists():
        return None
    fix = loadmat(path)["fixLocs"].astype(np.float32)
    if fix.shape != target_shape:
        return None
    return fix


def main() -> None:
    if not PRED_ROOT.exists():
        raise FileNotFoundError(f"Prediction root not found: {PRED_ROOT}")

    modes = ["plain", "inverted", "log_density_like", "inverted_log_density_like"]
    rows: list[dict[str, object]] = []

    categories = sorted([p for p in PRED_ROOT.iterdir() if p.is_dir()])

    for cat_dir in categories:
        files: list[Path] = []
        for pattern in PREDICTION_PATTERNS:
            files.extend(cat_dir.glob(pattern))
        files = sorted(files)
        if MAX_IMAGES_PER_CATEGORY > 0:
            files = files[:MAX_IMAGES_PER_CATEGORY]

        for pred_path in tqdm(files, desc=f"{cat_dir.name}"):
            pred = load_prediction_array(pred_path)

            for mode in modes:
                prob_map = to_prob_map(pred, mode)
                fix = load_fixation_map(cat_dir.name, pred_path.stem, prob_map.shape)
                if fix is None:
                    continue
                fix_density = density_from_fixation_map(fix, sigma=GAUSSIAN_SIGMA)

                rows.append(
                    {
                        "mode": mode,
                        "category": cat_dir.name,
                        "image": pred_path.stem,
                        "model_entropy": compute_entropy(prob_map),
                        "model_center_distance": center_distance(prob_map),
                        "model_num_peaks": count_peaks(prob_map),
                        "model_nss": nss(prob_map, fix),
                        "model_cc": cc(prob_map, fix_density),
                        "model_auc": auc_judd(prob_map, fix),
                    }
                )

    if not rows:
        raise RuntimeError("No rows computed in preprocessing sweep.")

    df = pd.DataFrame(rows)

    metric_cols = [
        "model_entropy",
        "model_center_distance",
        "model_num_peaks",
        "model_nss",
        "model_cc",
        "model_auc",
    ]

    overall = df.groupby("mode", as_index=False)[metric_cols].mean()
    overall.insert(1, "n_images", df.groupby("mode")["image"].count().values)

    per_category = df.groupby(["mode", "category"], as_index=False)[metric_cols].mean()

    OUT_OVERALL.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_OVERALL, index=False)
    per_category.to_csv(OUT_PER_CATEGORY, index=False)

    print(f"Saved: {OUT_OVERALL}")
    print(f"Saved: {OUT_PER_CATEGORY}")
    print("\nOverall summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
