"""Create a CAT2000 montage with one example image per category.

Examples:
    python scripts/plot_cat2000_examples_grid.py --stim-root C:/trainSet
    python scripts/plot_cat2000_examples_grid.py --stim-root C:/trainSet --sample-choice random --seed 7
    python scripts/plot_cat2000_examples_grid.py --stim-root C:/trainSet --overlay-fixations
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, zoom

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_stim_root(path_arg: Path) -> Path:
    """Accept either dataset root (.../CAT2000) or direct Stimuli path."""
    if not path_arg.exists():
        raise FileNotFoundError(
            f"Provided path not found: {path_arg}. "
            "Provide --stim-root as CAT2000 root or its Stimuli directory."
        )

    if (path_arg / "Stimuli").is_dir():
        return path_arg / "Stimuli"

    return path_arg


def resolve_fix_root(path_arg: Path, stim_root: Path, fix_root_arg: Path | None) -> Path:
    """Resolve fixation root from explicit argument or dataset-relative defaults."""
    if fix_root_arg is not None:
        if not fix_root_arg.exists():
            raise FileNotFoundError(f"Fixation root not found: {fix_root_arg}")
        if (fix_root_arg / "FIXATIONLOCS").is_dir():
            return fix_root_arg / "FIXATIONLOCS"
        return fix_root_arg

    if (path_arg / "FIXATIONLOCS").is_dir():
        return path_arg / "FIXATIONLOCS"

    candidate = stim_root.parent / "FIXATIONLOCS"
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        "Could not infer FIXATIONLOCS path. Provide --fix-root explicitly."
    )


def collect_example_images(stim_root: Path, sample_choice: str, seed: int) -> list[tuple[str, Path]]:
    """Return one example image path per category folder in CAT2000 Stimuli."""
    categories = sorted([p for p in stim_root.iterdir() if p.is_dir()])
    if not categories:
        raise RuntimeError(f"No category folders found in: {stim_root}")

    rng = random.Random(seed)
    examples: list[tuple[str, Path]] = []
    for category_dir in categories:
        images = sorted(
            [p for p in category_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )
        if not images:
            continue
        image_path = images[0] if sample_choice == "first" else rng.choice(images)
        examples.append((category_dir.name, image_path))

    if not examples:
        raise RuntimeError(f"No images found under category folders in: {stim_root}")

    return examples


def load_fixation_map(fix_path: Path) -> np.ndarray | None:
    """Load fixation map from MATLAB file, handling common CAT2000 key names."""
    try:
        mat = loadmat(fix_path)
    except Exception:
        return None

    preferred_keys = ["fixLocs", "fixLoc", "fixationMap", "fixations"]
    for key in preferred_keys:
        if key in mat:
            arr = np.asarray(mat[key], dtype=np.float32)
            if arr.ndim == 2:
                return arr

    best_arr: np.ndarray | None = None
    for value in mat.values():
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            continue
        if best_arr is None or value.size > best_arr.size:
            best_arr = value

    if best_arr is None:
        return None
    return np.asarray(best_arr, dtype=np.float32)


def resize_map_nearest(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize 2D map to target shape using nearest-neighbor interpolation."""
    target_h, target_w = target_shape
    if arr.shape == (target_h, target_w):
        return arr

    zoom_h = target_h / float(arr.shape[0])
    zoom_w = target_w / float(arr.shape[1])
    resized = zoom(arr, zoom=(zoom_h, zoom_w), order=0)
    return resized.astype(np.float32)


def fixation_density(
    fix_map: np.ndarray,
    target_shape: tuple[int, int],
    sigma: float,
) -> np.ndarray | None:
    """Convert fixation map into normalized smooth density for overlay."""
    fix_resized = resize_map_nearest(fix_map, target_shape)
    density = gaussian_filter(fix_resized, sigma=sigma)
    max_val = float(np.max(density))
    if max_val <= 0.0:
        return None
    density = density / max_val
    return density


def plot_examples(
    examples: list[tuple[str, Path]],
    output_path: Path,
    ncols: int,
    dpi: int,
    cell_width: float,
    cell_height: float,
    overlay_fixations: bool,
    fix_root: Path | None,
    overlay_sigma: float,
    overlay_alpha: float,
    overlay_cmap: str,
    label_fontsize: float,
    suptitle_fontsize: float,
    wspace: float,
    hspace: float,
) -> tuple[int, int]:
    """Plot image examples in a compact grid and save to disk."""
    n_images = len(examples)
    ncols = max(1, min(ncols, n_images))
    nrows = math.ceil(n_images / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * cell_width, nrows * cell_height),
        constrained_layout=False,
    )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Flatten regardless of subplot shape.
    flat_axes = np.atleast_1d(axes).ravel().tolist()

    overlays_used = 0
    overlays_missing = 0

    for ax, (category, img_path) in zip(flat_axes, examples):
        image = mpimg.imread(img_path)
        ax.imshow(image)

        if overlay_fixations and fix_root is not None:
            fix_path = fix_root / category / f"{img_path.stem}.mat"
            if fix_path.exists():
                fix_map = load_fixation_map(fix_path)
                if fix_map is not None:
                    density = fixation_density(
                        fix_map=fix_map,
                        target_shape=(image.shape[0], image.shape[1]),
                        sigma=overlay_sigma,
                    )
                    if density is not None:
                        ax.imshow(density, cmap=overlay_cmap, alpha=overlay_alpha)
                        overlays_used += 1
                    else:
                        overlays_missing += 1
                else:
                    overlays_missing += 1
            else:
                overlays_missing += 1

        ax.set_title(category, fontsize=label_fontsize)
        ax.axis("off")

    for ax in flat_axes[len(examples) :]:
        ax.axis("off")

    fig.suptitle("CAT2000: One Example Image per Category", fontsize=suptitle_fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return overlays_used, overlays_missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a matplotlib grid with one CAT2000 example image per category."
    )
    parser.add_argument(
        "--stim-root",
        type=Path,
        default=Path("data/Stimuli"),
        help="Path to CAT2000 Stimuli root (contains category folders).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cat2000_examples_grid.png"),
        help="Output image path for the generated grid.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=5,
        help="Number of columns in the grid (default: 5).",
    )
    parser.add_argument(
        "--sample-choice",
        choices=["first", "random"],
        default="first",
        help="Choose first or random image from each category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-choice random.",
    )
    parser.add_argument(
        "--overlay-fixations",
        action="store_true",
        help="Overlay smoothed fixation density from FIXATIONLOCS on each image.",
    )
    parser.add_argument(
        "--fix-root",
        type=Path,
        default=None,
        help="Path to FIXATIONLOCS root or dataset root containing FIXATIONLOCS.",
    )
    parser.add_argument(
        "--overlay-sigma",
        type=float,
        default=15.0,
        help="Gaussian sigma for fixation density smoothing.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Alpha transparency for fixation overlay.",
    )
    parser.add_argument(
        "--overlay-cmap",
        type=str,
        default="jet",
        help="Colormap for fixation overlay.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for publication-quality figure.",
    )
    parser.add_argument(
        "--cell-width",
        type=float,
        default=3.6,
        help="Width of each category cell in inches.",
    )
    parser.add_argument(
        "--cell-height",
        type=float,
        default=2.8,
        help="Height of each category cell in inches.",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=13.0,
        help="Category label font size.",
    )
    parser.add_argument(
        "--suptitle-fontsize",
        type=float,
        default=18.0,
        help="Figure title font size.",
    )
    parser.add_argument(
        "--wspace",
        type=float,
        default=0.03,
        help="Horizontal spacing between subplots (smaller = more compact).",
    )
    parser.add_argument(
        "--hspace",
        type=float,
        default=0.08,
        help="Vertical spacing between subplots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stim_root = resolve_stim_root(args.stim_root)
    fix_root: Path | None = None
    if args.overlay_fixations:
        fix_root = resolve_fix_root(path_arg=args.stim_root, stim_root=stim_root, fix_root_arg=args.fix_root)

    examples = collect_example_images(
        stim_root=stim_root,
        sample_choice=args.sample_choice,
        seed=args.seed,
    )
    overlays_used, overlays_missing = plot_examples(
        examples=examples,
        output_path=args.output,
        ncols=args.ncols,
        dpi=args.dpi,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        overlay_fixations=args.overlay_fixations,
        fix_root=fix_root,
        overlay_sigma=args.overlay_sigma,
        overlay_alpha=args.overlay_alpha,
        overlay_cmap=args.overlay_cmap,
        label_fontsize=args.label_fontsize,
        suptitle_fontsize=args.suptitle_fontsize,
        wspace=args.wspace,
        hspace=args.hspace,
    )
    print(f"Saved CAT2000 example grid: {args.output}")
    print(f"Stimuli root used: {stim_root}")
    print(f"Categories visualized: {len(examples)}")
    print(f"Sampling mode: {args.sample_choice}")
    if args.overlay_fixations:
        print(f"Fixation root used: {fix_root}")
        print(f"Overlays used: {overlays_used}")
        print(f"Overlays missing/failed: {overlays_missing}")


if __name__ == "__main__":
    main()
