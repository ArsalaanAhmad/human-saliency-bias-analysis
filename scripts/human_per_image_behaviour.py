from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import entropy
from tqdm import tqdm


FIX_ROOT = Path("data/FIXATIONLOCS")
OUTPUT_PATH = Path("outputs/human_per_image_behaviour.csv")

GAUSSIAN_SIGMA = 15
PEAK_THRESHOLD = 0.6
PEAK_NEIGHBORHOOD = 25


def compute_entropy(fixation_map: np.ndarray) -> float:
    prob = fixation_map.flatten().astype(np.float32)
    total = prob.sum()
    if total <= 0:
        return 0.0
    prob /= total
    return float(entropy(prob + 1e-8))


def center_distance(fixation_map: np.ndarray) -> float:
    h, w = fixation_map.shape
    cy, cx = h / 2, w / 2

    ys, xs = np.indices((h, w))
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    total = fixation_map.sum()
    if total <= 0:
        return float("nan")

    p = fixation_map / (total + 1e-8)
    return float((p * dists).sum())


def density_from_fixation_map(fixation_map: np.ndarray, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    density = gaussian_filter(fixation_map.astype(np.float32), sigma=sigma)
    max_value = float(density.max())
    if max_value > 0:
        density /= max_value
    return density


def count_peaks(density: np.ndarray, threshold: float = PEAK_THRESHOLD, neighborhood: int = PEAK_NEIGHBORHOOD) -> int:
    local_max = density == maximum_filter(density, size=neighborhood)
    peaks = local_max & (density >= threshold)
    return int(peaks.sum())


def iter_fixation_files():
    for category_dir in sorted([p for p in FIX_ROOT.iterdir() if p.is_dir()]):
        for mat_path in sorted(category_dir.glob("*.mat")):
            yield category_dir.name, mat_path


def main() -> None:
    if not FIX_ROOT.exists():
        raise RuntimeError(f"Fixation root not found: {FIX_ROOT}")

    rows = []
    skipped_empty = 0

    fixation_files = list(iter_fixation_files())
    if not fixation_files:
        raise RuntimeError(f"No .mat fixation files found under: {FIX_ROOT}")

    for category, mat_path in tqdm(fixation_files, desc="Human per-image metrics"):
        fixation_map = loadmat(mat_path)["fixLocs"].astype(np.float32)
        if fixation_map.sum() <= 0:
            skipped_empty += 1
            continue

        density = density_from_fixation_map(fixation_map)

        rows.append(
            {
                "category": category,
                "image": mat_path.stem,
                "human_entropy": compute_entropy(fixation_map),
                "human_center_distance": center_distance(fixation_map),
                "human_num_peaks": count_peaks(density),
            }
        )

    if not rows:
        raise RuntimeError("No valid per-image human rows were computed.")

    df = pd.DataFrame(rows).sort_values(["category", "image"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(df))
    print("Categories:", df["category"].nunique())
    print("Skipped empty fixation maps:", skipped_empty)


if __name__ == "__main__":
    main()
