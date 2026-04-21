from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, maximum_filter, zoom
from scipy.stats import entropy


# ============================================================
# CONFIG (edit these paths/settings)
# ============================================================
CATEGORY = "Satelite"  # CAT2000 category name in this dataset spelling
IMAGE_STEM = "001"     # e.g., 001, 003, 005

# CAT2000 roots (use either your trainSet paths or project-relative data paths)
STIM_ROOT = Path("C:/trainSet/Stimuli")
FIX_ROOT = Path("C:/trainSet/FIXATIONLOCS")

# Model prediction roots (each should contain category folders with maps)
DEEPGAZE_ROOT = Path("outputs/deepgaze_iie_cat2000")
SAMRESNET_ROOT = Path("outputs/samresnet_cat2000")
TRANSALNET_ROOT = Path("outputs/transalnet_cat2000")

# If True, also include center-bias panel.
INCLUDE_CENTER_BIAS = True

# Optional panel overlays and stats text.
OVERLAY_PEAK_MARKERS = True
SHOW_STATS_TEXT = True

# Saliency processing parameters.
HUMAN_DENSITY_SIGMA = 15
PEAK_THRESHOLD_RATIO = 0.60
PEAK_NEIGHBORHOOD = 25

# Plot/output settings.
FIGSIZE = (18, 10)
OUTPUT_FIG = Path("outputs/figures/satellite_single_image_multipanel.png")
OUTPUT_PDF = Path("outputs/figures/satellite_single_image_multipanel.pdf")
DPI = 300


# ============================================================
# Helpers
# ============================================================
@dataclass
class PanelData:
    name: str
    map_data: np.ndarray | None
    is_missing: bool = False
    missing_reason: str = ""


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_fixation_map(path: Path) -> np.ndarray:
    mat = loadmat(path)
    if "fixLocs" in mat:
        return np.asarray(mat["fixLocs"], dtype=np.float32)

    # Fallback: choose largest 2D array if key differs.
    best = None
    for value in mat.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if best is None or value.size > best.size:
                best = value

    if best is None:
        raise ValueError(f"Could not find 2D fixation map in {path}")
    return np.asarray(best, dtype=np.float32)


def resize_nearest(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr.astype(np.float32)

    zoom_h = target_shape[0] / float(arr.shape[0])
    zoom_w = target_shape[1] / float(arr.shape[1])
    out = zoom(arr, zoom=(zoom_h, zoom_w), order=0)
    return out.astype(np.float32)


def normalize_to_prob(arr: np.ndarray, is_log_density: bool = False) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if is_log_density:
        arr = np.exp(arr - float(np.max(arr)))
    else:
        arr = arr - float(np.min(arr))

    total = float(np.sum(arr))
    if total <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / total).astype(np.float32)


def density_from_fixation_map(fix_map: np.ndarray, sigma: float) -> np.ndarray:
    density = gaussian_filter(fix_map.astype(np.float32), sigma=sigma)
    max_val = float(np.max(density))
    if max_val > 0:
        density = density / max_val
    return density.astype(np.float32)


def find_prediction_file(root: Path, category: str, stem: str) -> Path | None:
    exts = [".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    # Preferred structure: root/category/stem.ext
    for ext in exts:
        candidate = root / category / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: root/stem.ext
    for ext in exts:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def load_prediction_map(path: Path, target_shape: Tuple[int, int], is_log_density: bool) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    arr = resize_nearest(arr, target_shape)
    return normalize_to_prob(arr, is_log_density=is_log_density)


def center_bias_map(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy, cx = h / 2.0, w / 2.0

    sigma = min(h, w) / 6.0
    gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return normalize_to_prob(gauss, is_log_density=False)


def compute_entropy(prob_map: np.ndarray) -> float:
    flat = prob_map.astype(np.float32).ravel()
    s = float(np.sum(flat))
    if s <= 0:
        return float("nan")
    p = flat / (s + 1e-8)
    return float(entropy(p + 1e-8))


def compute_center_distance(prob_map: np.ndarray) -> float:
    h, w = prob_map.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy, cx = h / 2.0, w / 2.0

    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    p = normalize_to_prob(prob_map, is_log_density=False)
    return float(np.sum(p * d))


def peak_mask(prob_map: np.ndarray, threshold_ratio: float, neighborhood: int) -> np.ndarray:
    max_val = float(np.max(prob_map))
    if max_val <= 0:
        return np.zeros_like(prob_map, dtype=bool)

    norm = prob_map / max_val
    local_max = norm == maximum_filter(norm, size=neighborhood)
    return local_max & (norm >= threshold_ratio)


def compute_num_peaks(prob_map: np.ndarray, threshold_ratio: float, neighborhood: int) -> int:
    return int(np.count_nonzero(peak_mask(prob_map, threshold_ratio, neighborhood)))


def panel_stats_text(prob_map: np.ndarray) -> str:
    return (
        f"peaks: {compute_num_peaks(prob_map, PEAK_THRESHOLD_RATIO, PEAK_NEIGHBORHOOD)}\n"
        f"entropy: {compute_entropy(prob_map):.3f}\n"
        f"centre dist: {compute_center_distance(prob_map):.1f}"
    )


def build_model_panel(name: str, root: Path, category: str, stem: str, shape: Tuple[int, int], is_log_density: bool) -> PanelData:
    if not root.exists():
        return PanelData(name=name, map_data=None, is_missing=True, missing_reason=f"root not found: {root}")

    pred_path = find_prediction_file(root=root, category=category, stem=stem)
    if pred_path is None:
        return PanelData(name=name, map_data=None, is_missing=True, missing_reason=f"map not found for {category}/{stem}")

    try:
        pred_map = load_prediction_map(path=pred_path, target_shape=shape, is_log_density=is_log_density)
    except Exception as exc:
        return PanelData(name=name, map_data=None, is_missing=True, missing_reason=str(exc))

    return PanelData(name=name, map_data=pred_map, is_missing=False)


def draw_panel(ax: plt.Axes, panel: PanelData, show_peaks: bool, show_stats: bool) -> None:
    ax.set_title(panel.name, fontsize=12)
    ax.axis("off")

    if panel.is_missing or panel.map_data is None:
        ax.imshow(np.zeros((256, 256), dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        ax.text(
            0.5,
            0.5,
            "Missing map",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            transform=ax.transAxes,
        )
        if panel.missing_reason:
            ax.set_xlabel(panel.missing_reason, fontsize=8)
        return

    ax.imshow(panel.map_data, cmap="magma")

    if show_peaks:
        peaks = peak_mask(panel.map_data, threshold_ratio=PEAK_THRESHOLD_RATIO, neighborhood=PEAK_NEIGHBORHOOD)
        ys, xs = np.where(peaks)
        if ys.size > 0:
            ax.scatter(xs, ys, s=12, c="#00E5FF", marker="x", linewidths=0.9)

    if show_stats:
        ax.set_xlabel(panel_stats_text(panel.map_data), fontsize=9)


def main() -> None:
    stim_path = STIM_ROOT / CATEGORY / f"{IMAGE_STEM}.jpg"
    fix_path = FIX_ROOT / CATEGORY / f"{IMAGE_STEM}.mat"

    if not stim_path.exists():
        raise FileNotFoundError(f"Image not found: {stim_path}")
    if not fix_path.exists():
        raise FileNotFoundError(f"Fixation map not found: {fix_path}")

    image_rgb = load_rgb_image(stim_path)
    target_shape = (image_rgb.shape[0], image_rgb.shape[1])

    fix_map = load_fixation_map(fix_path)
    fix_map = resize_nearest(fix_map, target_shape)
    human_map = normalize_to_prob(density_from_fixation_map(fix_map, sigma=HUMAN_DENSITY_SIGMA), is_log_density=False)

    panels: List[PanelData] = [
        PanelData(name="Original image", map_data=None, is_missing=False),
        PanelData(name="Human attention map", map_data=human_map, is_missing=False),
        build_model_panel(
            name="DeepGaze IIE",
            root=DEEPGAZE_ROOT,
            category=CATEGORY,
            stem=IMAGE_STEM,
            shape=target_shape,
            is_log_density=True,
        ),
        build_model_panel(
            name="SAM-ResNet",
            root=SAMRESNET_ROOT,
            category=CATEGORY,
            stem=IMAGE_STEM,
            shape=target_shape,
            is_log_density=False,
        ),
        build_model_panel(
            name="TranSalNet",
            root=TRANSALNET_ROOT,
            category=CATEGORY,
            stem=IMAGE_STEM,
            shape=target_shape,
            is_log_density=False,
        ),
    ]

    if INCLUDE_CENTER_BIAS:
        panels.append(PanelData(name="Centre-bias baseline", map_data=center_bias_map(target_shape), is_missing=False))

    n_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=FIGSIZE)
    flat_axes = np.atleast_1d(axes).ravel().tolist()

    for idx, panel in enumerate(panels):
        ax = flat_axes[idx]
        if panel.name == "Original image":
            ax.imshow(image_rgb)
            ax.set_title(panel.name, fontsize=12)
            ax.axis("off")
            if SHOW_STATS_TEXT:
                ax.set_xlabel("source image", fontsize=9)
        else:
            draw_panel(ax, panel, show_peaks=OVERLAY_PEAK_MARKERS, show_stats=SHOW_STATS_TEXT)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Single-image Saliency Comparison ({CATEGORY} {IMAGE_STEM})",
        fontsize=14,
        y=0.99,
    )

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.02, 1.0, 0.96])
    fig.savefig(OUTPUT_FIG, dpi=DPI)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)

    print(f"Saved: {OUTPUT_FIG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
