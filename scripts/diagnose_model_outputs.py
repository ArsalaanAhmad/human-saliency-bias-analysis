"""
Diagnostic script for SAM-ResNet (PNG noise-floor peak inflation)
and TranSalNet (near-uniform activation / low CC).

Run from the project root with D:\ mounted:
    python scripts/diagnose_model_outputs.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

SAMRESNET_ROOT  = Path("D:/outputs/sam-resnet_cat2000")
TRANSALNET_ROOT = Path("D:/outputs/transalnet_cat2000_50")
EXTENSIONS      = [".npy", ".png", ".jpg", ".jpeg"]


def find_first_file(root: Path, extensions: list[str]) -> Path | None:
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ext in extensions:
            matches = sorted(cat_dir.glob(f"*{ext}"))
            if matches:
                return matches[0]
    return None


def load_raw(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def prediction_to_prob_map(arr: np.ndarray, is_log_density: bool) -> np.ndarray:
    arr = arr.astype(np.float32)
    if is_log_density:
        p = np.exp(arr - arr.max())
    else:
        p = arr - arr.min()
    total = p.sum()
    if total <= 0:
        return np.zeros_like(p)
    return (p / total).astype(np.float32)


def count_peaks(arr: np.ndarray, floor_pct: float,
                threshold_ratio: float = 0.6, neighborhood: int = 25) -> int:
    clipped = np.where(arr < floor_pct * arr.max(), 0.0, arr)
    max_val = clipped.max()
    if max_val <= 0:
        return 0
    norm = clipped / max_val
    local_max = norm == maximum_filter(norm, size=neighborhood)
    return int((local_max & (norm >= threshold_ratio)).sum())


def print_array_stats(label: str, arr: np.ndarray) -> None:
    flat = arr.flatten()
    pcts = np.percentile(flat, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    frac_above_90pct_max = float((flat >= 0.9 * flat.max()).mean())
    print(f"  [{label}]")
    print(f"    shape       : {arr.shape}  dtype: {arr.dtype}")
    print(f"    min / max   : {flat.min():.4f} / {flat.max():.4f}")
    print(f"    mean / std  : {flat.mean():.4f} / {flat.std():.4f}")
    print(f"    unique vals : {len(np.unique(flat))}")
    print(f"    percentiles :")
    labels = ["0.1%", "  1%", "  5%", " 25%", " 50%", " 75%", " 95%", " 99%", "99.9%"]
    for lbl, v in zip(labels, pcts):
        print(f"      {lbl}  {v:.4f}")
    print(f"    frac >= 90% of max : {frac_above_90pct_max:.4f}  (1.0 = fully uniform)")


def run_peak_battery(arr: np.ndarray) -> None:
    for floor in (0.00, 0.01, 0.05):
        n = count_peaks(arr, floor_pct=floor)
        tag = "(current)" if floor == 0.00 else ("(proposed fix)" if floor == 0.01 else "(aggressive)")
        print(f"    floor={floor:.0%} → {n:4d} peaks  {tag}")


def diagnose(model_label: str, root: Path) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {model_label}")
    print(sep)

    if not root.exists():
        print(f"  SKIP: {root} does not exist (D:\\ not mounted?)")
        return

    path = find_first_file(root, EXTENSIONS)
    if path is None:
        print(f"  SKIP: no matching files found under {root}")
        return

    print(f"  file: {path}")
    raw = load_raw(path)

    if raw.ndim == 3:
        print(f"\n  WARNING: array has 3 dimensions {raw.shape} — expected 2D (H, W)")
        if raw.shape[0] == 1:
            print("  Squeezing dim 0 for further analysis.")
            squeezed = raw.squeeze(0)
        else:
            print("  Cannot auto-squeeze; showing raw stats only.")
            print_array_stats("RAW (3D)", raw)
            return
    else:
        squeezed = raw

    print()
    print_array_stats("RAW", squeezed)

    print()
    print("  --- Peak counts on RAW array ---")
    run_peak_battery(squeezed)

    print()
    print("  --- prob_map (is_log_density=False) ---")
    prob_plain = prediction_to_prob_map(squeezed, is_log_density=False)
    print_array_stats("prob (log=F)", prob_plain)
    print("  Peak counts:")
    run_peak_battery(prob_plain)

    print()
    print("  --- prob_map (is_log_density=True) ---")
    prob_log = prediction_to_prob_map(squeezed, is_log_density=True)
    print_array_stats("prob (log=T)", prob_log)
    print("  Peak counts:")
    run_peak_battery(prob_log)


if __name__ == "__main__":
    diagnose("SAM-ResNet  (PNG noise-floor peak inflation)", SAMRESNET_ROOT)
    diagnose("TranSalNet  (near-uniform activation / low CC)", TRANSALNET_ROOT)
    print()
