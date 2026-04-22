from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, maximum_filter


# ------------------------------------------------------------
# Helpers: loading and normalization
# ------------------------------------------------------------
def load_saliency_image(path: Path) -> np.ndarray:
    """Load a saliency map from PNG/JPG, .npy, or .mat and return float32 2D array."""
    if not path.exists():
        raise FileNotFoundError(f"Saliency image not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif suffix == ".mat":
        mat = loadmat(path)
        if "fixLocs" in mat and isinstance(mat["fixLocs"], np.ndarray):
            arr = np.asarray(mat["fixLocs"], dtype=np.float32)
        else:
            arr = _largest_2d_array(mat)
    else:
        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Saliency map must be 2D after loading. Got shape {arr.shape} from {path}")

    return arr


def load_optional_map(path: Path) -> np.ndarray:
    """Load an optional map from image, .npy, or .mat for visualization only."""
    if not path.exists():
        raise FileNotFoundError(f"Optional map not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif suffix == ".mat":
        mat = loadmat(path)
        if "fixLocs" in mat and isinstance(mat["fixLocs"], np.ndarray):
            arr = np.asarray(mat["fixLocs"], dtype=np.float32)
        else:
            arr = _largest_2d_array(mat)
    else:
        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Optional map must be 2D after loading. Got shape {arr.shape} from {path}")

    return arr


def _largest_2d_array(mat_dict: dict) -> np.ndarray:
    best = None
    for value in mat_dict.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if best is None or value.size > best.size:
                best = value

    if best is None:
        raise ValueError("Could not find any 2D array in .mat file")

    return np.asarray(best, dtype=np.float32)


def normalize_saliency_map(arr: np.ndarray, is_log_density: bool = False) -> np.ndarray:
    """
    Convert map to a stable float32 saliency/probability map.

    Steps:
    1) remove NaN/Inf
    2) shift minimum to zero
    3) normalize by sum (probability map)
    """
    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if is_log_density:
        x = np.exp(x - float(np.max(x)))
    else:
        x = x - float(np.min(x))

    total = float(np.sum(x))
    if total <= 0:
        return np.zeros_like(x, dtype=np.float32)

    return (x / total).astype(np.float32)


# ------------------------------------------------------------
# Helpers: peak detection
# ------------------------------------------------------------
def detect_peaks(
    saliency_map: np.ndarray,
    threshold_ratio: float = 0.6,
    neighborhood: int = 25,
    blur_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect peaks using the current method:
    - optional Gaussian smoothing first
    - normalize by map max
    - local maxima via maximum_filter
    - threshold relative to max

    Returns:
    - peak_mask (bool 2D)
    - peak_coords as Nx2 array of (y, x)
    """
    work = np.asarray(saliency_map, dtype=np.float32)
    if blur_sigma > 0:
        work = gaussian_filter(work, sigma=blur_sigma)

    max_val = float(np.max(work))
    if max_val <= 0:
        empty_mask = np.zeros_like(work, dtype=bool)
        return empty_mask, np.empty((0, 2), dtype=np.int32)

    norm = work / max_val
    local_max = norm == maximum_filter(norm, size=int(neighborhood), mode="nearest")
    peak_mask = local_max & (norm >= float(threshold_ratio))

    ys, xs = np.where(peak_mask)
    coords = np.column_stack((ys, xs)).astype(np.int32)
    return peak_mask, coords


def format_coords(coords: np.ndarray) -> str:
    if coords.size == 0:
        return "[]"
    tuples = [f"({int(y)}, {int(x)})" for y, x in coords]
    return "[" + ", ".join(tuples) + "]"


# ------------------------------------------------------------
# Helpers: plotting
# ------------------------------------------------------------
def plot_peak_overlay(
    ax: plt.Axes,
    background_map: np.ndarray,
    coords: np.ndarray,
    title: str,
    cmap: str = "magma",
) -> None:
    """Plot a saliency map and overlay peak coordinates."""
    ax.imshow(background_map, cmap=cmap)
    if coords.size > 0:
        ax.scatter(coords[:, 1], coords[:, 0], s=18, c="#00E5FF", marker="x", linewidths=0.9)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_debug_figure(
    saliency_map: np.ndarray,
    results: List[dict],
    output_path: Path,
    stimulus_image: np.ndarray | None = None,
    human_map: np.ndarray | None = None,
) -> None:
    """Build and save the multi-panel diagnostic figure."""
    panels: List[tuple[str, np.ndarray | None, np.ndarray | None, bool]] = []
    # tuple = (title, image, coords, is_overlay)

    if stimulus_image is not None:
        panels.append(("Stimulus (optional)", stimulus_image, None, False))

    panels.append(("SAM-ResNet saliency map", saliency_map, None, False))

    if human_map is not None:
        panels.append(("Human map (optional)", human_map, None, False))

    for item in results:
        panels.append((item["title"], saliency_map, item["coords"], True))

    n_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.8 * nrows))
    flat_axes = np.atleast_1d(axes).ravel().tolist()

    for i, (title, image, coords, is_overlay) in enumerate(panels):
        ax = flat_axes[i]
        if image is None:
            ax.axis("off")
            continue

        if is_overlay:
            plot_peak_overlay(ax=ax, background_map=image, coords=coords if coords is not None else np.empty((0, 2)), title=title)
        else:
            if image.ndim == 2:
                ax.imshow(image, cmap="magma")
            else:
                ax.imshow(image)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("SAM-ResNet Peak Detection Sanity Check", fontsize=14, y=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image SAM-ResNet peak-detection sanity check"
    )
    parser.add_argument("--sam-map", required=True, type=Path, help="Path to one saliency map (PNG/JPG/.npy/.mat)")
    parser.add_argument("--stimulus", type=Path, default=None, help="Optional original stimulus image path")
    parser.add_argument("--human-map", type=Path, default=None, help="Optional human fixation-density map path (image/.npy/.mat)")
    parser.add_argument("--output", type=Path, default=Path("outputs/debug_sam_peak_detection.png"), help="Output diagnostic figure path")

    parser.add_argument("--threshold-ratio", type=float, default=0.6, help="Relative threshold for baseline peak detection")
    parser.add_argument("--neighborhood", type=int, default=25, help="Neighborhood size for local maxima")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian sigma for smoothed variant")
    parser.add_argument("--strict-threshold", type=float, default=0.7, help="Stricter threshold variant")
    parser.add_argument(
        "--log-density",
        action="store_true",
        help="Interpret input saliency map as log-density (e.g., DeepGaze .npy) and exponentiate before normalization",
    )
    parser.add_argument(
        "--larger-neighborhoods",
        type=int,
        nargs="+",
        default=[35, 45],
        help="Alternative larger neighborhood sizes to test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1-3) Load and normalize SAM map
    sam_raw = load_saliency_image(args.sam_map)
    sam_map = normalize_saliency_map(sam_raw, is_log_density=args.log_density)

    # Optional context panels
    stimulus_image = None
    if args.stimulus is not None:
        stim_img = Image.open(args.stimulus).convert("RGB")
        stimulus_image = np.asarray(stim_img)

    human_map = None
    if args.human_map is not None:
        human_raw = load_optional_map(args.human_map)
        human_map = normalize_saliency_map(human_raw)

    # 4-5) Peak detection variants
    results: List[dict] = []

    # Current settings
    _, coords_current = detect_peaks(
        saliency_map=sam_map,
        threshold_ratio=args.threshold_ratio,
        neighborhood=args.neighborhood,
        blur_sigma=0.0,
    )
    results.append(
        {
            "key": "current",
            "title": f"Current: thr={args.threshold_ratio}, n={args.neighborhood} | peaks={len(coords_current)}",
            "coords": coords_current,
        }
    )

    # Light smoothing + current threshold/neighborhood
    _, coords_smooth = detect_peaks(
        saliency_map=sam_map,
        threshold_ratio=args.threshold_ratio,
        neighborhood=args.neighborhood,
        blur_sigma=args.smooth_sigma,
    )
    results.append(
        {
            "key": "smoothed",
            "title": f"Smoothed: sigma={args.smooth_sigma}, thr={args.threshold_ratio}, n={args.neighborhood} | peaks={len(coords_smooth)}",
            "coords": coords_smooth,
        }
    )

    # Larger neighborhoods (compute all requested, show the largest in figure)
    larger_results: List[dict] = []
    for nbh in args.larger_neighborhoods:
        _, coords_large = detect_peaks(
            saliency_map=sam_map,
            threshold_ratio=args.threshold_ratio,
            neighborhood=nbh,
            blur_sigma=0.0,
        )
        larger_results.append(
            {
                "key": f"large_n_{nbh}",
                "title": f"Larger n={nbh}: thr={args.threshold_ratio} | peaks={len(coords_large)}",
                "coords": coords_large,
                "nbh": nbh,
            }
        )

    # Show the strictest neighborhood (largest value) as requested in figure.
    larger_results = sorted(larger_results, key=lambda x: x["nbh"])
    if larger_results:
        results.append(larger_results[-1])

    # Stricter threshold
    _, coords_strict = detect_peaks(
        saliency_map=sam_map,
        threshold_ratio=args.strict_threshold,
        neighborhood=args.neighborhood,
        blur_sigma=0.0,
    )
    results.append(
        {
            "key": "strict",
            "title": f"Stricter thr={args.strict_threshold}, n={args.neighborhood} | peaks={len(coords_strict)}",
            "coords": coords_strict,
        }
    )

    # 6) Print counts and coordinates
    print("\n=== SAM Peak Detection Sanity Check ===")
    print(f"Input SAM map: {args.sam_map}")
    print(f"Map shape: {sam_map.shape}")

    print("\n[Current]")
    print(f"num_peaks = {len(coords_current)}")
    print(f"coordinates = {format_coords(coords_current)}")

    print("\n[Smoothed]")
    print(f"num_peaks = {len(coords_smooth)}")
    print(f"coordinates = {format_coords(coords_smooth)}")

    print("\n[Larger neighborhood variants]")
    for item in larger_results:
        coords = item["coords"]
        print(f"n={item['nbh']} -> num_peaks = {len(coords)}")
        print(f"coordinates = {format_coords(coords)}")

    print("\n[Stricter threshold]")
    print(f"num_peaks = {len(coords_strict)}")
    print(f"coordinates = {format_coords(coords_strict)}")

    # 7-9) Plot and save diagnostic figure
    plot_debug_figure(
        saliency_map=sam_map,
        results=results,
        output_path=args.output,
        stimulus_image=stimulus_image,
        human_map=human_map,
    )

    print(f"\nSaved figure: {args.output}")


if __name__ == "__main__":
    main()
