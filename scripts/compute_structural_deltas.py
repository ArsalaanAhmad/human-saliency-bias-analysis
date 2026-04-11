from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


# Editable mapping from file stem to model name used in outputs.
# Example: "samresnet_per_image_behaviour": "SAM-ResNet"
MODEL_NAME_MAP = {
    "deepgazeiie_per_image_behaviour": "deepgazeiie",
    "samresnet_per_image_behaviour": "samresnet",
    "transalnet_per_image_behaviour": "transalnet",
}

MERGE_KEYS = ["category", "image"]
MODEL_REQUIRED_COLUMNS = [
    "category",
    "image",
    "model_entropy",
    "model_center_distance",
    "model_num_peaks",
    "model_nss",
    "model_cc",
    "model_auc",
]
HUMAN_REQUIRED_COLUMNS = [
    "category",
    "image",
    "human_entropy",
    "human_center_distance",
    "human_num_peaks",
]
DELTA_COLUMNS = [
    "delta_entropy",
    "delta_center_distance",
    "delta_num_peaks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge model per-image behaviour CSV(s) with human per-image behaviour CSV, "
            "compute structural deltas, and save per-model and combined summaries."
        )
    )
    parser.add_argument(
        "--human-csv",
        default="outputs/human_per_image_behaviour.csv",
        help="Path to human per-image behaviour CSV.",
    )
    parser.add_argument(
        "--model-csvs",
        nargs="*",
        default=None,
        help="Optional explicit model CSV paths. If omitted, --model-glob is used.",
    )
    parser.add_argument(
        "--model-glob",
        default="outputs/*_per_image_behaviour.csv",
        help="Glob used to discover model CSVs when --model-csvs is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for output CSV files.",
    )
    return parser.parse_args()


def ensure_file_exists(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"[ERROR] {label} not found: {path}")
        return False
    if not path.is_file():
        print(f"[ERROR] {label} is not a file: {path}")
        return False
    return True


def validate_columns(df: pd.DataFrame, required: Sequence[str], csv_path: Path) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in {csv_path}: {missing}")
        return False
    return True


def infer_model_name(model_csv_path: Path) -> str:
    stem = model_csv_path.stem
    if stem in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[stem]
    return stem.replace("_per_image_behaviour", "")


def discover_model_csvs(
    human_csv: Path,
    model_csvs: Sequence[str] | None,
    model_glob: str,
) -> List[Path]:
    if model_csvs:
        candidates = [Path(p) for p in model_csvs]
    else:
        candidates = list(Path().glob(model_glob))

    resolved_human = human_csv.resolve()
    filtered: List[Path] = []
    seen: set[Path] = set()

    for path in candidates:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path

        if resolved == resolved_human:
            continue
        if "human" in path.stem.lower():
            continue
        if path.suffix.lower() != ".csv":
            continue
        if resolved in seen:
            continue

        filtered.append(path)
        seen.add(resolved)

    return sorted(filtered)


def category_image_set(df: pd.DataFrame) -> set[Tuple[str, str]]:
    return set(zip(df["category"].astype(str), df["image"].astype(str)))


def print_missing_pairs_summary(
    model_name: str,
    in_model_not_human: set[Tuple[str, str]],
    in_human_not_model: set[Tuple[str, str]],
    sample_n: int = 5,
) -> None:
    print(f"\n[{model_name}] Missing merge keys summary")
    print(f"  - In model but missing in human: {len(in_model_not_human)}")
    if in_model_not_human:
        sample = sorted(in_model_not_human)[:sample_n]
        print(f"    Sample: {sample}")

    print(f"  - In human but missing in model: {len(in_human_not_model)}")
    if in_human_not_model:
        sample = sorted(in_human_not_model)[:sample_n]
        print(f"    Sample: {sample}")


def main() -> None:
    args = parse_args()

    human_csv = Path(args.human_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ensure_file_exists(human_csv, "Human CSV"):
        raise SystemExit(1)

    human_df = pd.read_csv(human_csv)
    if not validate_columns(human_df, HUMAN_REQUIRED_COLUMNS, human_csv):
        raise SystemExit(1)

    human_df = human_df[HUMAN_REQUIRED_COLUMNS].copy()

    model_paths = discover_model_csvs(human_csv, args.model_csvs, args.model_glob)
    if not model_paths:
        print("[ERROR] No model CSVs found to process.")
        print("Hint: pass --model-csvs explicitly or adjust --model-glob.")
        raise SystemExit(1)

    all_per_image: List[pd.DataFrame] = []
    all_per_category: List[pd.DataFrame] = []
    all_overall: List[pd.DataFrame] = []

    print("\nProcessing model files:")
    for p in model_paths:
        print(f"  - {p}")

    for model_csv in model_paths:
        if not ensure_file_exists(model_csv, "Model CSV"):
            continue

        model_df = pd.read_csv(model_csv)
        if not validate_columns(model_df, MODEL_REQUIRED_COLUMNS, model_csv):
            continue

        model_name = infer_model_name(model_csv)

        model_key_set = category_image_set(model_df)
        human_key_set = category_image_set(human_df)

        in_model_not_human = model_key_set - human_key_set
        in_human_not_model = human_key_set - model_key_set

        merged = model_df.merge(human_df, on=MERGE_KEYS, how="inner")

        if merged.empty:
            print(f"\n[WARN] {model_name}: no matched rows after merge on {MERGE_KEYS}.")
            print_missing_pairs_summary(model_name, in_model_not_human, in_human_not_model)
            continue

        merged["model_name"] = model_name
        merged["delta_entropy"] = (merged["model_entropy"] - merged["human_entropy"]).abs()
        merged["delta_center_distance"] = (
            merged["model_center_distance"] - merged["human_center_distance"]
        ).abs()
        merged["delta_num_peaks"] = (merged["model_num_peaks"] - merged["human_num_peaks"]).abs()

        per_image_columns = [
            "category",
            "image",
            "model_name",
            "model_entropy",
            "model_center_distance",
            "model_num_peaks",
            "model_nss",
            "model_cc",
            "model_auc",
            "human_entropy",
            "human_center_distance",
            "human_num_peaks",
            "delta_entropy",
            "delta_center_distance",
            "delta_num_peaks",
        ]
        per_image = merged[per_image_columns].copy()

        per_category = (
            per_image.groupby("category", as_index=False)[DELTA_COLUMNS]
            .mean()
            .sort_values("category")
        )
        per_category.insert(0, "model_name", model_name)

        overall = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "matched_rows": int(len(per_image)),
                    "mean_delta_entropy": float(per_image["delta_entropy"].mean()),
                    "mean_delta_center_distance": float(per_image["delta_center_distance"].mean()),
                    "mean_delta_num_peaks": float(per_image["delta_num_peaks"].mean()),
                }
            ]
        )

        # Save per-model files.
        per_image_path = output_dir / f"{model_name}_human_delta_per_image.csv"
        per_category_path = output_dir / f"{model_name}_human_delta_per_category_mean.csv"
        overall_path = output_dir / f"{model_name}_human_delta_overall_mean.csv"

        per_image.to_csv(per_image_path, index=False)
        per_category.to_csv(per_category_path, index=False)
        overall.to_csv(overall_path, index=False)

        # Keep for combined all-model outputs.
        all_per_image.append(per_image)
        all_per_category.append(per_category)
        all_overall.append(overall)

        print(f"\n[{model_name}] Summary")
        print(f"  - Matched rows: {len(per_image)}")
        print_missing_pairs_summary(model_name, in_model_not_human, in_human_not_model)
        print(
            "  - Overall mean deltas: "
            f"entropy={overall.loc[0, 'mean_delta_entropy']:.6f}, "
            f"center_distance={overall.loc[0, 'mean_delta_center_distance']:.6f}, "
            f"num_peaks={overall.loc[0, 'mean_delta_num_peaks']:.6f}"
        )

    if not all_per_image:
        print("\n[ERROR] No model files were successfully processed.")
        raise SystemExit(1)

    combined_per_image = pd.concat(all_per_image, ignore_index=True)
    combined_per_category = pd.concat(all_per_category, ignore_index=True)
    combined_overall = pd.concat(all_overall, ignore_index=True)

    combined_per_image_path = output_dir / "all_models_human_delta_per_image.csv"
    combined_per_category_path = output_dir / "all_models_human_delta_per_category_mean.csv"
    combined_overall_path = output_dir / "all_models_human_delta_overall_mean.csv"

    combined_per_image.to_csv(combined_per_image_path, index=False)
    combined_per_category.to_csv(combined_per_category_path, index=False)
    combined_overall.to_csv(combined_overall_path, index=False)

    print("\nSaved combined outputs:")
    print(f"  - {combined_per_image_path}")
    print(f"  - {combined_per_category_path}")
    print(f"  - {combined_overall_path}")


if __name__ == "__main__":
    main()
