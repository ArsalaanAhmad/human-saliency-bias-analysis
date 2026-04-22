from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, maximum_filter, zoom


# ============================================================
# CONFIG: edit this section
# ============================================================
# Category display name -> folder name in your CAT2000 data/model outputs.
# CAT2000 in this repo uses "Satelite" spelling on disk.
CATEGORY_FOLDER_MAP: Dict[str, str] = {
    "Satellite": "Satelite",
    "Jumbled": "Jumbled",
    "Social": "Social",
    "Pattern": "Pattern",
    "Sketch": "Sketch",
}

# Which image IDs to render per category (use numeric strings like "021").
IMAGE_IDS_BY_CATEGORY: Dict[str, List[str]] = {
    "Satellite": ["015"],
    "Sketch": ["017"],
}

# Data roots (first existing root in each candidate list will be used).
STIM_ROOT_CANDIDATES = [
    Path("data/Stimuli"),
    Path("C:/trainSet/Stimuli"),
]
FIX_LOCS_ROOT_CANDIDATES = [
    Path("data/FIXATIONLOCS"),
    Path("C:/trainSet/FIXATIONLOCS"),
]

# Model roots (first existing root per model is used).
MODEL_ROOT_CANDIDATES: Dict[str, List[Path]] = {
    "DeepGaze IIE": [
        Path("D:/outputs/deepgaze_iie_cat2000_50"),
        Path("D:/outputs/deepgaze_iie_cat2000"),
        Path("outputs/deepgaze_iie_cat2000"),
        Path("outputs/deepgazeiie_cat2000"),
    ],
    "SAM-ResNet": [
        Path("D:/outputs/sam-resnet_cat2000"),
        Path("D:/outputs/samresnet_cat2000"),
        Path("outputs/samresnet_cat2000"),
    ],
    "TranSalNet": [
        Path("D:/outputs/transalnet_cat2000_50"),
        Path("D:/outputs/transalnet_cat2000"),
        Path("outputs/transalnet_cat2000"),
    ],
}

# If True, include a centre Gaussian prior panel.
INCLUDE_CENTRE_PRIOR = True

# If True, overlay saliency maps on top of the original image for all map panels.
OVERLAY_SALIENCY_ON_ORIGINAL = True
OVERLAY_ALPHA = 0.55

# Optional overlays.
OVERLAY_FIXATION_POINTS = False
OVERLAY_PEAK_MARKERS = True

# Peak detector settings.
PEAK_THRESHOLD_RATIO = 0.6
PEAK_NEIGHBORHOOD = 25

# Human fixation-density smoothing.
HUMAN_DENSITY_SIGMA = 15

# Output folder.
OUTPUT_ROOT = Path("outputs/figures/qualitative_cases")
DPI = 260
FIGSIZE = (18, 10)


# ============================================================
# Helpers
# ============================================================
@dataclass
class PanelSpec:
    title: str
    map_data: np.ndarray | None
    missing_reason: str = ""


def resolve_existing_root(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_category_dir(root: Path, folder_name: str) -> Path | None:
    if not root.exists():
        return None

    wanted = folder_name.casefold()
    aliases = {wanted}
    if wanted == "satelite":
        aliases.add("satellite")
    if wanted == "satellite":
        aliases.add("satelite")

    for child in root.iterdir():
        if child.is_dir() and child.name.casefold() in aliases:
            return child
    return None


def find_image_file(folder: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def load_fixation_map(mat_path: Path) -> np.ndarray:
    mat = loadmat(mat_path)
    if "fixLocs" in mat and isinstance(mat["fixLocs"], np.ndarray):
        return np.asarray(mat["fixLocs"], dtype=np.float32)

    best = None
    for value in mat.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if best is None or value.size > best.size:
                best = value

    if best is None:
        raise ValueError(f"Could not find 2D fixation map in {mat_path}")
    return np.asarray(best, dtype=np.float32)


def resize_nearest(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr.astype(np.float32)

    zoom_h = target_shape[0] / float(arr.shape[0])
    zoom_w = target_shape[1] / float(arr.shape[1])
    out = zoom(arr, zoom=(zoom_h, zoom_w), order=0)
    return out.astype(np.float32)


def normalize_to_prob(arr: np.ndarray, is_log_density: bool = False) -> np.ndarray:
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


def human_density_from_fixations(fix_map: np.ndarray, sigma: float) -> np.ndarray:
    density = gaussian_filter(fix_map.astype(np.float32), sigma=sigma)
    return normalize_to_prob(density, is_log_density=False)


def find_prediction_file(root: Path, category_folder: str, stem: str) -> Path | None:
    exts = (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # Common: root/category/stem.ext
    for ext in exts:
        p = root / category_folder / f"{stem}{ext}"
        if p.exists():
            return p

    # Fallback aliases for Satellite/Satelite folder naming
    cat_aliases = [category_folder]
    if category_folder.casefold() == "satelite":
        cat_aliases.append("Satellite")
    if category_folder.casefold() == "satellite":
        cat_aliases.append("Satelite")

    for cat_name in cat_aliases:
        cat_dir = resolve_category_dir(root, cat_name)
        if cat_dir is None:
            continue
        for ext in exts:
            p = cat_dir / f"{stem}{ext}"
            if p.exists():
                return p

    # Fallback: root/stem.ext
    for ext in exts:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def load_prediction_map(path: Path, target_shape: Tuple[int, int], is_log_density: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        arr = load_gray_image(path)

    arr = resize_nearest(arr, target_shape)
    return normalize_to_prob(arr, is_log_density=is_log_density)


def center_prior_map(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy, cx = h / 2.0, w / 2.0
    sigma = min(h, w) / 6.0
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return normalize_to_prob(g, is_log_density=False)


def peak_mask(prob_map: np.ndarray, threshold_ratio: float, neighborhood: int) -> np.ndarray:
    max_val = float(np.max(prob_map))
    if max_val <= 0:
        return np.zeros_like(prob_map, dtype=bool)

    norm = prob_map / max_val
    local_max = norm == maximum_filter(norm, size=neighborhood)
    return local_max & (norm >= threshold_ratio)


def draw_panel(
    ax: plt.Axes,
    panel: PanelSpec,
    image_rgb: np.ndarray,
    fixation_points: np.ndarray | None,
) -> None:
    ax.set_title(panel.title, fontsize=12)
    ax.axis("off")

    if panel.map_data is None:
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

    if OVERLAY_SALIENCY_ON_ORIGINAL:
        ax.imshow(image_rgb)
        ax.imshow(panel.map_data, cmap="magma", alpha=OVERLAY_ALPHA)
    else:
        ax.imshow(panel.map_data, cmap="magma")

    if OVERLAY_FIXATION_POINTS and fixation_points is not None and fixation_points.size > 0:
        # fixation_points is Nx2 as (y, x)
        ax.scatter(
            fixation_points[:, 1],
            fixation_points[:, 0],
            s=3,
            c="#7CFC00",
            alpha=0.45,
            marker="o",
            linewidths=0,
        )

    if OVERLAY_PEAK_MARKERS:
        peaks = peak_mask(panel.map_data, threshold_ratio=PEAK_THRESHOLD_RATIO, neighborhood=PEAK_NEIGHBORHOOD)
        ys, xs = np.where(peaks)
        if ys.size > 0:
            ax.scatter(xs, ys, s=12, c="#00E5FF", marker="x", linewidths=0.9)


def render_case(
    category_display: str,
    category_folder: str,
    image_id: str,
    stim_root: Path,
    fix_root: Path,
    model_roots: Dict[str, Path | None],
) -> None:
    stim_dir = resolve_category_dir(stim_root, category_folder)
    fix_dir = resolve_category_dir(fix_root, category_folder)
    if stim_dir is None:
        print(f"[WARN] Missing stimulus category folder: {category_display} ({category_folder})")
        return
    if fix_dir is None:
        print(f"[WARN] Missing fixation category folder: {category_display} ({category_folder})")
        return

    stim_path = find_image_file(stim_dir, image_id)
    fix_mat_path = fix_dir / f"{image_id}.mat"

    if stim_path is None:
        print(f"[WARN] Missing stimulus image: {category_display}/{image_id}")
        return
    if not fix_mat_path.exists():
        print(f"[WARN] Missing fixation MAT: {category_display}/{image_id}")
        return

    image_rgb = load_rgb_image(stim_path)
    target_shape = (image_rgb.shape[0], image_rgb.shape[1])

    fix_map = load_fixation_map(fix_mat_path)
    fix_map = resize_nearest(fix_map, target_shape)
    human_map = human_density_from_fixations(fix_map, sigma=HUMAN_DENSITY_SIGMA)

    fixation_points = None
    if OVERLAY_FIXATION_POINTS:
        ys, xs = np.where(fix_map > 0)
        if ys.size > 0:
            fixation_points = np.column_stack((ys, xs)).astype(np.int32)

    # DeepGaze output is typically log-density when loaded from .npy.
    # For image outputs we still use same normalization path (min-shift + sum via normalize_to_prob).
    model_specs = [
        ("DeepGaze IIE", True),
        ("SAM-ResNet", False),
        ("TranSalNet", False),
    ]

    panels: List[PanelSpec] = [
        PanelSpec(title="Original image", map_data=None),
        PanelSpec(title="Human fixation-density", map_data=human_map),
    ]

    # Load model maps robustly across .npy and image formats.
    for model_name, default_logdensity in model_specs:
        root = model_roots.get(model_name)
        if root is None:
            panels.append(PanelSpec(title=model_name, map_data=None, missing_reason="model root missing"))
            continue

        pred_path = find_prediction_file(root, category_folder, image_id)
        if pred_path is None:
            panels.append(PanelSpec(title=model_name, map_data=None, missing_reason=f"map not found for {category_folder}/{image_id}"))
            continue

        # DeepGaze .npy are log-density; image exports are regular intensity.
        is_log = bool(default_logdensity and pred_path.suffix.lower() == ".npy")

        try:
            pred_map = load_prediction_map(pred_path, target_shape=target_shape, is_log_density=is_log)
            panels.append(PanelSpec(title=model_name, map_data=pred_map))
        except Exception as exc:
            panels.append(PanelSpec(title=model_name, map_data=None, missing_reason=str(exc)))

    if INCLUDE_CENTRE_PRIOR:
        panels.append(PanelSpec(title="Centre Gaussian prior", map_data=center_prior_map(target_shape)))

    n_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=FIGSIZE)
    flat_axes = np.atleast_1d(axes).ravel().tolist()

    for idx, panel in enumerate(panels):
        ax = flat_axes[idx]
        if panel.title == "Original image":
            ax.imshow(image_rgb)
            ax.set_title(panel.title, fontsize=12)
            ax.axis("off")
        else:
            draw_panel(ax=ax, panel=panel, image_rgb=image_rgb, fixation_points=fixation_points)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"{category_display} {image_id} - Qualitative Saliency Comparison", fontsize=14, y=0.99)

    out_dir = OUTPUT_ROOT / category_display
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{category_display}_{image_id}_comparison.png"

    fig.tight_layout(rect=[0.01, 0.02, 1.0, 0.96])
    fig.savefig(out_file, dpi=DPI)
    plt.close(fig)

    print(f"Saved: {out_file}")


def main() -> None:
    stim_root = resolve_existing_root(STIM_ROOT_CANDIDATES)
    fix_root = resolve_existing_root(FIX_LOCS_ROOT_CANDIDATES)

    if stim_root is None:
        raise FileNotFoundError(f"No stimulus root found in: {STIM_ROOT_CANDIDATES}")
    if fix_root is None:
        raise FileNotFoundError(f"No fixation-locations root found in: {FIX_LOCS_ROOT_CANDIDATES}")

    model_roots: Dict[str, Path | None] = {}
    for model_name, candidates in MODEL_ROOT_CANDIDATES.items():
        model_roots[model_name] = resolve_existing_root(candidates)
        print(f"{model_name} root: {model_roots[model_name] if model_roots[model_name] is not None else 'MISSING'}")

    for category_display, image_ids in IMAGE_IDS_BY_CATEGORY.items():
        if category_display not in CATEGORY_FOLDER_MAP:
            print(f"[WARN] Missing CATEGORY_FOLDER_MAP entry for category: {category_display}")
            continue

        category_folder = CATEGORY_FOLDER_MAP[category_display]

        for image_id in image_ids:
            # Normalize IDs to 3 digits for CAT2000 naming.
            stem = f"{int(image_id):03d}" if image_id.isdigit() else image_id
            render_case(
                category_display=category_display,
                category_folder=category_folder,
                image_id=stem,
                stim_root=stim_root,
                fix_root=fix_root,
                model_roots=model_roots,
            )


if __name__ == "__main__":
    main()
