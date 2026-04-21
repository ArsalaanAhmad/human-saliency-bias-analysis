from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter


# -------------------------
# configuration
# -------------------------

HUMAN_CSV = Path("outputs/human_per_image_behaviour.csv")
OUTPUT_DIR = Path("outputs")

# If model folders contain category subfolders, keep this True.
# Expected structure?: pred_root/category/image.ext
USE_CATEGORY_SUBDIRS = True

# Settings for sensitivity analysis
THRESHOLD_RATIOS = [0.5, 0.6, 0.7]
NEIGHBORHOODS = [15, 25, 35]
PROGRESS_EVERY_ROWS = 300

# If True, also save one CSV file per setting (threshold_ratio, neighborhood)
# under outputs/peak_sensitivity_by_setting.
SAVE_SPLIT_BY_SETTING = True

# Model configuration list.
# extension can be ".npy", ".jpg", ".png", etc. only None ONLY to auto-detect common types.
MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "model_name": "deepgazeiie",
        "pred_root": "D:/outputs/deepgaze_iie_cat2000_50",
        "extension": ".npy",
        "is_log_density": True,
    },
    {
        "model_name": "samresnet",
        "pred_root": "D:/outputs/sam-resnet_cat2000",
        "extension": ".jpg",
        "is_log_density": True,
    },
    {
        "model_name": "transalnet",
        "pred_root": "D:/outputs/transalnet_cat2000_50",
        "extension": ".jpg",
        "is_log_density": False,
    },
]


COMMON_EXTENSIONS = [".npy", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def load_human_peaks(human_csv: Path) -> pd.DataFrame:
    """Load human per-image peak counts with required columns."""
    if not human_csv.exists():
        raise FileNotFoundError(f"Human CSV not found: {human_csv}")

    # Read image IDs as strings to preserve any zero-padding (e.g., 001).
    human_df = pd.read_csv(human_csv, dtype={"category": str, "image": str})
    required = ["category", "image", "human_num_peaks"]
    missing = [col for col in required if col not in human_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in human CSV {human_csv}: {missing}")

    out = human_df[required].copy()
    out["category"] = out["category"].astype(str).str.strip()
    out["image"] = out["image"].astype(str).map(normalize_image_key)
    out["human_num_peaks"] = pd.to_numeric(out["human_num_peaks"], errors="coerce")
    out = out.dropna(subset=["human_num_peaks"]).reset_index(drop=True)
    return out


def normalize_image_key(value: str) -> str:
    """Normalize image IDs so keys match whether CSV has stems or filenames."""
    text = str(value).strip()
    if not text:
        return text
    return Path(text).stem


def list_category_dirs(pred_root: Path) -> list[Path]:
    return sorted([p for p in pred_root.iterdir() if p.is_dir()])


def collect_prediction_index(
    pred_root: Path,
    extension: str | None,
    use_category_subdirs: bool,
) -> tuple[dict[tuple[str, str], Path], int]:
    """
    Build index: (category, image_stem) -> file path.

    Returns index and duplicate-key count.
    """
    index: dict[tuple[str, str], Path] = {}
    duplicate_count = 0

    if not pred_root.exists():
        return index, duplicate_count

    if extension is None:
        exts = COMMON_EXTENSIONS
    else:
        ext = extension.lower()
        exts = [ext] if ext.startswith(".") else [f".{ext}"]

    if use_category_subdirs:
        categories = list_category_dirs(pred_root)
        for cat_dir in categories:
            category = cat_dir.name
            files: list[Path] = []
            for ext in exts:
                files.extend(cat_dir.glob(f"*{ext}"))

            for path in sorted(files):
                key = (str(category), str(path.stem))
                if key in index:
                    duplicate_count += 1
                    continue
                index[key] = path
    else:
        files: list[Path] = []
        for ext in exts:
            files.extend(pred_root.glob(f"*{ext}"))

        for path in sorted(files):
            # If no category dirs, category cannot be inferred robustly; use empty string.
            key = ("", str(path.stem))
            if key in index:
                duplicate_count += 1
                continue
            index[key] = path

    return index, duplicate_count


def load_npy_prediction_map(pred_path: Path) -> np.ndarray:
    """Load an ndarray prediction map stored in .npy format."""
    return np.load(pred_path).astype(np.float32)


def load_image_prediction_map(pred_path: Path) -> np.ndarray:
    """Load an image-based prediction map and convert to grayscale float32."""
    image = Image.open(pred_path).convert("L")
    return np.asarray(image, dtype=np.float32)


def load_prediction_array(pred_path: Path) -> np.ndarray:
    """Dispatch prediction loading based on file extension."""
    suffix = pred_path.suffix.lower()
    if suffix == ".npy":
        return load_npy_prediction_map(pred_path)

    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return load_image_prediction_map(pred_path)

    raise ValueError(f"Unsupported prediction extension: {pred_path}")


def prediction_to_prob_map(prediction: np.ndarray, is_log_density: bool) -> np.ndarray:
    """Convert saved model output to probability map for structural analysis."""
    pred = prediction.astype(np.float32)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    if pred.ndim != 2:
        # Most saved saliency maps should be 2D. If a map has extra singleton
        # dimensions, squeeze them; otherwise fail fast for clearly bad inputs.
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise ValueError(f"Prediction must be 2D after squeeze, got shape {prediction.shape}")

    if is_log_density:
        prob_map = np.exp(pred - float(np.max(pred)))
    else:
        pred = pred - float(np.min(pred))
        prob_map = pred

    total = float(np.sum(prob_map))
    if total <= 0:
        return np.zeros_like(prob_map, dtype=np.float32)
    return (prob_map / total).astype(np.float32)


def count_peaks(prob_map: np.ndarray, threshold_ratio: float, neighborhood: int) -> int:
    max_value = float(prob_map.max())
    if max_value <= 0:
        return 0

    norm = prob_map / max_value
    local_max = norm == maximum_filter(norm, size=neighborhood)
    peaks = local_max & (norm >= threshold_ratio)
    return int(peaks.sum())


def compute_model_peak_sensitivity(
    model_name: str,
    pred_index: dict[tuple[str, str], Path],
    is_log_density: bool,
    human_df: pd.DataFrame,
    threshold_ratios: list[float],
    neighborhoods: list[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Compute per-image model peak counts for all sensitivity settings.

    Returns per-image dataframe and diagnostic counts.
    """
    rows: list[dict[str, Any]] = []
    cache_prob_maps: dict[Path, np.ndarray] = {}

    missing_prediction_rows = 0
    load_failures = 0
    processed_rows = 0
    total_rows = int(len(human_df))

    for _, human_row in human_df.iterrows():
        processed_rows += 1
        category = str(human_row["category"])
        image = str(human_row["image"])

        pred_path = resolve_prediction_path(pred_index=pred_index, category=category, image=image)
        if pred_path is None:
            missing_prediction_rows += 1
            continue

        if pred_path not in cache_prob_maps:
            try:
                pred_array = load_prediction_array(pred_path)
                cache_prob_maps[pred_path] = prediction_to_prob_map(pred_array, is_log_density=is_log_density)
            except Exception:
                load_failures += 1
                continue

        prob_map = cache_prob_maps[pred_path]

        for threshold_ratio in threshold_ratios:
            for neighborhood in neighborhoods:
                rows.append(
                    {
                        "model_name": model_name,
                        "category": category,
                        "image": image,
                        "threshold_ratio": float(threshold_ratio),
                        "neighborhood": int(neighborhood),
                        "model_num_peaks": count_peaks(
                            prob_map,
                            threshold_ratio=threshold_ratio,
                            neighborhood=neighborhood,
                        ),
                    }
                )

        if PROGRESS_EVERY_ROWS > 0 and (
            processed_rows % PROGRESS_EVERY_ROWS == 0 or processed_rows == total_rows
        ):
            print(
                "    progress: "
                f"{processed_rows}/{total_rows} human rows | "
                f"matched images={len(cache_prob_maps)} | "
                f"missing={missing_prediction_rows} | "
                f"load_failures={load_failures}",
                flush=True,
            )

    per_image_df = pd.DataFrame(rows)
    diagnostics = {
        "total_human_rows": total_rows,
        "processed_human_rows": processed_rows,
        "missing_prediction_rows": missing_prediction_rows,
        "load_failures": load_failures,
        "matched_unique_images": int(per_image_df[["category", "image"]].drop_duplicates().shape[0])
        if not per_image_df.empty
        else 0,
    }
    return per_image_df, diagnostics


def resolve_prediction_path(
    pred_index: dict[tuple[str, str], Path],
    category: str,
    image: str,
) -> Path | None:
    """Resolve prediction path while tolerating numeric zero-padding mismatches."""
    # 1) exact key
    key = (category, image)
    path = pred_index.get(key)
    if path is not None:
        return path

    # 2) stripped numeric (001 -> 1)
    if image.isdigit():
        stripped = str(int(image))
        path = pred_index.get((category, stripped))
        if path is not None:
            return path

        # 3) common zero-padded variants (1 -> 001 / 0001)
        for width in (3, 4):
            padded = image.zfill(width)
            path = pred_index.get((category, padded))
            if path is not None:
                return path

    return None


def summarize_key_overlap(
    pred_index: dict[tuple[str, str], Path],
    human_df: pd.DataFrame,
    sample_n: int = 5,
) -> dict[str, Any]:
    """Return overlap stats and examples for debugging key mismatches."""
    human_keys = {(str(r["category"]), str(r["image"])) for _, r in human_df.iterrows()}
    pred_keys = set(pred_index.keys())
    overlap = human_keys & pred_keys
    missing = sorted(human_keys - pred_keys)

    return {
        "human_key_count": len(human_keys),
        "prediction_key_count": len(pred_keys),
        "overlap_count": len(overlap),
        "sample_missing_human_keys": missing[:sample_n],
        "sample_prediction_keys": sorted(pred_keys)[:sample_n],
    }


def aggregate_peak_outputs(per_image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_category = (
        per_image_df.groupby(["model_name", "category", "threshold_ratio", "neighborhood"], as_index=False)[
            "model_num_peaks"
        ]
        .mean()
        .rename(columns={"model_num_peaks": "mean_model_num_peaks"})
    )

    overall = (
        per_image_df.groupby(["model_name", "threshold_ratio", "neighborhood"], as_index=False)["model_num_peaks"]
        .mean()
        .rename(columns={"model_num_peaks": "mean_model_num_peaks"})
    )
    return per_category, overall


def merge_with_human_and_compute_delta(per_image_df: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
    merged = per_image_df.merge(human_df, on=["category", "image"], how="left")
    merged["human_num_peaks"] = pd.to_numeric(merged["human_num_peaks"], errors="coerce")
    merged["delta_num_peaks"] = (merged["model_num_peaks"] - merged["human_num_peaks"]).abs()
    return merged


def aggregate_delta_outputs(delta_per_image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_category = (
        delta_per_image_df.groupby(["model_name", "category", "threshold_ratio", "neighborhood"], as_index=False)[
            "delta_num_peaks"
        ]
        .mean()
        .rename(columns={"delta_num_peaks": "mean_delta_num_peaks"})
    )

    overall = (
        delta_per_image_df.groupby(["model_name", "threshold_ratio", "neighborhood"], as_index=False)["delta_num_peaks"]
        .mean()
        .rename(columns={"delta_num_peaks": "mean_delta_num_peaks"})
    )
    return per_category, overall


def ranking_signature(overall_delta_df: pd.DataFrame) -> pd.DataFrame:
    """Model ordering by mean delta for each setting (lower is better)."""
    rows: list[dict[str, Any]] = []

    settings = (
        overall_delta_df[["threshold_ratio", "neighborhood"]]
        .drop_duplicates()
        .sort_values(["threshold_ratio", "neighborhood"])
    )
    if settings.empty:
        return pd.DataFrame(
            columns=[
                "threshold_ratio",
                "neighborhood",
                "model_order",
                "ordering_stable_vs_baseline",
            ]
        )

    baseline_setting = tuple(settings.iloc[0].tolist())
    baseline_slice = overall_delta_df[
        (overall_delta_df["threshold_ratio"] == baseline_setting[0])
        & (overall_delta_df["neighborhood"] == baseline_setting[1])
    ]
    baseline_order = tuple(
        baseline_slice.sort_values("mean_delta_num_peaks")["model_name"].astype(str).tolist()
    )

    for _, setting in settings.iterrows():
        threshold_ratio = float(setting["threshold_ratio"])
        neighborhood = int(setting["neighborhood"])

        setting_slice = overall_delta_df[
            (overall_delta_df["threshold_ratio"] == threshold_ratio)
            & (overall_delta_df["neighborhood"] == neighborhood)
        ]
        model_order = tuple(
            setting_slice.sort_values("mean_delta_num_peaks")["model_name"].astype(str).tolist()
        )

        rows.append(
            {
                "threshold_ratio": threshold_ratio,
                "neighborhood": neighborhood,
                "model_order": " > ".join(model_order),
                "ordering_stable_vs_baseline": model_order == baseline_order,
            }
        )

    return pd.DataFrame(rows)


def high_mismatch_category_signature(
    per_category_delta_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Category stability summary across settings.

    Uses category delta averaged across models for each setting.
    """
    rows: list[dict[str, Any]] = []

    agg = (
        per_category_delta_df.groupby(["threshold_ratio", "neighborhood", "category"], as_index=False)[
            "mean_delta_num_peaks"
        ]
        .mean()
        .rename(columns={"mean_delta_num_peaks": "category_delta_across_models"})
    )

    settings = (
        agg[["threshold_ratio", "neighborhood"]]
        .drop_duplicates()
        .sort_values(["threshold_ratio", "neighborhood"])
    )
    if settings.empty:
        return pd.DataFrame(
            columns=[
                "threshold_ratio",
                "neighborhood",
                "top_categories",
                "same_high_mismatch_as_baseline",
                "overlap_with_baseline",
            ]
        )

    baseline_setting = tuple(settings.iloc[0].tolist())
    baseline_slice = agg[
        (agg["threshold_ratio"] == baseline_setting[0])
        & (agg["neighborhood"] == baseline_setting[1])
    ]
    baseline_top = baseline_slice.sort_values("category_delta_across_models", ascending=False).head(top_k)[
        "category"
    ].astype(str).tolist()
    baseline_top_set = set(baseline_top)

    for _, setting in settings.iterrows():
        threshold_ratio = float(setting["threshold_ratio"])
        neighborhood = int(setting["neighborhood"])

        setting_slice = agg[
            (agg["threshold_ratio"] == threshold_ratio)
            & (agg["neighborhood"] == neighborhood)
        ]
        top_categories = (
            setting_slice.sort_values("category_delta_across_models", ascending=False)
            .head(top_k)["category"]
            .astype(str)
            .tolist()
        )
        top_set = set(top_categories)
        overlap = len(top_set & baseline_top_set)

        rows.append(
            {
                "threshold_ratio": threshold_ratio,
                "neighborhood": neighborhood,
                "top_categories": " | ".join(top_categories),
                "same_high_mismatch_as_baseline": top_set == baseline_top_set,
                "overlap_with_baseline": overlap,
            }
        )

    return pd.DataFrame(rows)


def save_model_outputs(
    model_name: str,
    per_image_peaks: pd.DataFrame,
    per_category_peaks: pd.DataFrame,
    overall_peaks: pd.DataFrame,
    per_image_delta: pd.DataFrame,
    per_category_delta: pd.DataFrame,
    overall_delta: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save per-model outputs for peak counts and peak deltas."""
    per_image_peaks.to_csv(output_dir / f"{model_name}_peak_sensitivity_per_image.csv", index=False)
    per_category_peaks.to_csv(output_dir / f"{model_name}_peak_sensitivity_per_category_mean.csv", index=False)
    overall_peaks.to_csv(output_dir / f"{model_name}_peak_sensitivity_overall_mean.csv", index=False)

    per_image_delta.to_csv(output_dir / f"{model_name}_delta_peak_sensitivity_per_image.csv", index=False)
    per_category_delta.to_csv(output_dir / f"{model_name}_delta_peak_sensitivity_per_category_mean.csv", index=False)
    overall_delta.to_csv(output_dir / f"{model_name}_delta_peak_sensitivity_overall_mean.csv", index=False)


def _setting_tag(threshold_ratio: float, neighborhood: int) -> str:
    threshold_tag = str(float(threshold_ratio)).replace(".", "p")
    return f"thr{threshold_tag}_n{int(neighborhood)}"


def save_outputs_by_setting(
    df: pd.DataFrame,
    base_filename: str,
    output_root: Path,
) -> None:
    """Save one CSV per (threshold_ratio, neighborhood) setting."""
    if df.empty:
        return

    required_cols = {"threshold_ratio", "neighborhood"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{base_filename}: missing required columns {required_cols}")

    out_dir = output_root / "peak_sensitivity_by_setting" / base_filename
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = (
        df[["threshold_ratio", "neighborhood"]]
        .drop_duplicates()
        .sort_values(["threshold_ratio", "neighborhood"])
    )

    for _, row in settings.iterrows():
        threshold_ratio = float(row["threshold_ratio"])
        neighborhood = int(row["neighborhood"])
        subset = df[
            (df["threshold_ratio"] == threshold_ratio)
            & (df["neighborhood"] == neighborhood)
        ].copy()

        out_name = f"{base_filename}_{_setting_tag(threshold_ratio, neighborhood)}.csv"
        subset.to_csv(out_dir / out_name, index=False)


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    human_df = load_human_peaks(HUMAN_CSV)

    all_per_image_peaks: list[pd.DataFrame] = []
    all_per_category_peaks: list[pd.DataFrame] = []
    all_overall_peaks: list[pd.DataFrame] = []

    all_per_image_delta: list[pd.DataFrame] = []
    all_per_category_delta: list[pd.DataFrame] = []
    all_overall_delta: list[pd.DataFrame] = []

    print("Human rows:", len(human_df))

    for cfg in MODEL_CONFIGS:
        model_name = str(cfg["model_name"])
        pred_root = Path(str(cfg["pred_root"]))
        extension = cfg.get("extension")
        is_log_density = bool(cfg.get("is_log_density", False))

        pred_index, duplicate_count = collect_prediction_index(
            pred_root=pred_root,
            extension=extension,
            use_category_subdirs=USE_CATEGORY_SUBDIRS,
        )

        print(
            f"\n[{model_name}] indexing from {pred_root} (ext={extension}, log_density={is_log_density})",
            flush=True,
        )

        if not pred_root.exists():
            print(f"[WARN] {model_name}: prediction root missing, skipping: {pred_root}")
            continue
        if not pred_index:
            print(f"[WARN] {model_name}: no prediction files found under {pred_root}, skipping")
            continue

        per_image_peaks, diagnostics = compute_model_peak_sensitivity(
            model_name=model_name,
            pred_index=pred_index,
            is_log_density=is_log_density,
            human_df=human_df,
            threshold_ratios=THRESHOLD_RATIOS,
            neighborhoods=NEIGHBORHOODS,
        )

        if per_image_peaks.empty:
            print(f"[WARN] {model_name}: no matched rows after loading predictions, skipping")
            overlap_debug = summarize_key_overlap(pred_index=pred_index, human_df=human_df, sample_n=8)
            print(
                "       Key-overlap debug: "
                f"human={overlap_debug['human_key_count']}, "
                f"pred={overlap_debug['prediction_key_count']}, "
                f"overlap={overlap_debug['overlap_count']}"
            )
            print(f"       Sample missing human keys: {overlap_debug['sample_missing_human_keys']}")
            print(f"       Sample prediction keys: {overlap_debug['sample_prediction_keys']}")
            continue

        per_category_peaks, overall_peaks = aggregate_peak_outputs(per_image_peaks)

        per_image_delta = merge_with_human_and_compute_delta(per_image_peaks, human_df)
        per_category_delta, overall_delta = aggregate_delta_outputs(per_image_delta)

        save_model_outputs(
            model_name=model_name,
            per_image_peaks=per_image_peaks,
            per_category_peaks=per_category_peaks,
            overall_peaks=overall_peaks,
            per_image_delta=per_image_delta,
            per_category_delta=per_category_delta,
            overall_delta=overall_delta,
            output_dir=output_dir,
        )

        all_per_image_peaks.append(per_image_peaks)
        all_per_category_peaks.append(per_category_peaks)
        all_overall_peaks.append(overall_peaks)

        all_per_image_delta.append(per_image_delta)
        all_per_category_delta.append(per_category_delta)
        all_overall_delta.append(overall_delta)

        print(f"\n[{model_name}]")
        print(f"  Indexed predictions: {len(pred_index)} (duplicates skipped: {duplicate_count})")
        print(f"  Total human rows considered: {diagnostics['total_human_rows']}")
        print(f"  Missing prediction rows vs human: {diagnostics['missing_prediction_rows']}")
        print(f"  Prediction load failures: {diagnostics['load_failures']}")
        print(f"  Matched unique images: {diagnostics['matched_unique_images']}")

    if not all_per_image_peaks:
        raise RuntimeError("No model data was processed. Check MODEL_CONFIGS and file paths.")

    combined_per_image_peaks = pd.concat(all_per_image_peaks, ignore_index=True)
    combined_per_category_peaks = pd.concat(all_per_category_peaks, ignore_index=True)
    combined_overall_peaks = pd.concat(all_overall_peaks, ignore_index=True)

    combined_per_image_delta = pd.concat(all_per_image_delta, ignore_index=True)
    combined_per_category_delta = pd.concat(all_per_category_delta, ignore_index=True)
    combined_overall_delta = pd.concat(all_overall_delta, ignore_index=True)

    combined_per_image_peaks.to_csv(output_dir / "all_models_peak_sensitivity_per_image.csv", index=False)
    combined_per_category_peaks.to_csv(output_dir / "all_models_peak_sensitivity_per_category_mean.csv", index=False)
    combined_overall_peaks.to_csv(output_dir / "all_models_peak_sensitivity_overall_mean.csv", index=False)

    combined_per_image_delta.to_csv(output_dir / "all_models_delta_peak_sensitivity_per_image.csv", index=False)
    combined_per_category_delta.to_csv(
        output_dir / "all_models_delta_peak_sensitivity_per_category_mean.csv",
        index=False,
    )
    combined_overall_delta.to_csv(output_dir / "all_models_delta_peak_sensitivity_overall_mean.csv", index=False)

    ordering_summary = ranking_signature(combined_overall_delta)
    mismatch_summary = high_mismatch_category_signature(combined_per_category_delta, top_k=5)
    compact_summary = ordering_summary.merge(
        mismatch_summary,
        on=["threshold_ratio", "neighborhood"],
        how="outer",
    ).sort_values(["threshold_ratio", "neighborhood"]).reset_index(drop=True)

    compact_summary.to_csv(output_dir / "peak_sensitivity_compact_summary.csv", index=False)

    if SAVE_SPLIT_BY_SETTING:
        save_outputs_by_setting(
            combined_per_image_peaks,
            "all_models_peak_sensitivity_per_image",
            output_dir,
        )
        save_outputs_by_setting(
            combined_per_category_peaks,
            "all_models_peak_sensitivity_per_category_mean",
            output_dir,
        )
        save_outputs_by_setting(
            combined_overall_peaks,
            "all_models_peak_sensitivity_overall_mean",
            output_dir,
        )
        save_outputs_by_setting(
            combined_per_image_delta,
            "all_models_delta_peak_sensitivity_per_image",
            output_dir,
        )
        save_outputs_by_setting(
            combined_per_category_delta,
            "all_models_delta_peak_sensitivity_per_category_mean",
            output_dir,
        )
        save_outputs_by_setting(
            combined_overall_delta,
            "all_models_delta_peak_sensitivity_overall_mean",
            output_dir,
        )

    print("\nSaved combined outputs:")
    print(f"  - {output_dir / 'all_models_peak_sensitivity_per_image.csv'}")
    print(f"  - {output_dir / 'all_models_peak_sensitivity_per_category_mean.csv'}")
    print(f"  - {output_dir / 'all_models_peak_sensitivity_overall_mean.csv'}")
    print(f"  - {output_dir / 'all_models_delta_peak_sensitivity_per_image.csv'}")
    print(f"  - {output_dir / 'all_models_delta_peak_sensitivity_per_category_mean.csv'}")
    print(f"  - {output_dir / 'all_models_delta_peak_sensitivity_overall_mean.csv'}")
    print(f"  - {output_dir / 'peak_sensitivity_compact_summary.csv'}")
    if SAVE_SPLIT_BY_SETTING:
        print(f"  - {output_dir / 'peak_sensitivity_by_setting'}")

    print("\nCompact summary table:")
    print(compact_summary.to_string(index=False))


if __name__ == "__main__":
    main()
