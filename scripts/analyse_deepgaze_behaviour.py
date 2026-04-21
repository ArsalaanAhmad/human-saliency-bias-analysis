import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
from scipy.stats import entropy
from scipy.ndimage import maximum_filter

from metrics import auc_judd, cc, density_from_fixation_map, nss

FIX_ROOT = Path("data/FIXATIONLOCS")
GAUSSIAN_SIGMA = 15
MAX_IMAGES_PER_CATEGORY = int(os.getenv("MAX_IMAGES_PER_CATEGORY", "50"))

# DeepGaze IIE example:
# MODEL_NAME = "deepgaze_iie"
# PRED_ROOT = Path("outputs/deepgaze_iie_cat2000")
# PREDICTION_IS_LOG_DENSITY = True

# SAM-ResNet example:
MODEL_NAME = "samresnet"
PRED_ROOT = Path("D:/outputs/sam-resnet_cat2000")
PREDICTION_IS_LOG_DENSITY = False

print("Prediction root:", PRED_ROOT)
print("Exists:", PRED_ROOT.exists())
print("First few entries:", list(PRED_ROOT.iterdir())[:5])

def compute_entropy(prob_map):
    p = prob_map.flatten().astype(np.float32)
    p = p / (p.sum() + 1e-8)
    return float(entropy(p + 1e-8))

def center_distance(prob_map):
    h, w = prob_map.shape
    cy, cx = h / 2, w / 2

    ys, xs = np.indices((h, w))
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    p = prob_map / (prob_map.sum() + 1e-8)
    return float((p * dists).sum())

def count_peaks(prob_map, threshold_ratio=0.6, neighborhood=25):
    if prob_map.max() == 0:
        return 0

    norm = prob_map / prob_map.max()
    local_max = (norm == maximum_filter(norm, size=neighborhood))
    peaks = local_max & (norm >= threshold_ratio)
    return int(peaks.sum())


def prediction_to_prob_map(prediction):
    prediction = prediction.astype(np.float32)
    if PREDICTION_IS_LOG_DENSITY:
        prob_map = np.exp(prediction - prediction.max())
    else:
        prediction = prediction - prediction.min()
        prob_map = prediction

    total = prob_map.sum()
    if total <= 0:
        return np.zeros_like(prob_map, dtype=np.float32)
    return (prob_map / total).astype(np.float32)


def load_prediction_array(prediction_path):
    suffix = prediction_path.suffix.lower()
    if suffix == ".npy":
        return np.load(prediction_path).astype(np.float32)

    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        image = Image.open(prediction_path).convert("L")
        return np.asarray(image, dtype=np.float32)

    raise ValueError(f"Unsupported prediction file extension: {prediction_path}")


def load_fixation_map(category, image_name, target_shape):
    fixation_path = FIX_ROOT / category / f"{image_name}.mat"
    if not fixation_path.exists():
        return None

    fixation_map = loadmat(fixation_path)["fixLocs"].astype(np.float32)
    if fixation_map.shape != target_shape:
        raise ValueError(
            f"Fixation map shape {fixation_map.shape} does not match prediction shape {target_shape} for {fixation_path}"
        )
    return fixation_map

rows = []

prediction_patterns = ["*.npy", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

for cat_dir in sorted([p for p in PRED_ROOT.iterdir() if p.is_dir()]):
    prediction_files = []
    for pattern in prediction_patterns:
        prediction_files.extend(cat_dir.glob(pattern))

    sorted_predictions = sorted(prediction_files)
    if MAX_IMAGES_PER_CATEGORY > 0:
        sorted_predictions = sorted_predictions[:MAX_IMAGES_PER_CATEGORY]

    for pred_path in tqdm(sorted_predictions, desc=f"Analysing {cat_dir.name}"):

        prediction = load_prediction_array(pred_path)
        prob_map = prediction_to_prob_map(prediction)
        fixation_map = load_fixation_map(cat_dir.name, pred_path.stem, prob_map.shape)

        if fixation_map is None:
            print(f"Skipping {pred_path}: missing fixation map")
            continue

        fixation_density = density_from_fixation_map(fixation_map, sigma=GAUSSIAN_SIGMA)

        rows.append({
            "category": cat_dir.name,
            "image": pred_path.stem,
            "model_entropy": compute_entropy(prob_map),
            "model_center_distance": center_distance(prob_map),
            "model_num_peaks": count_peaks(prob_map),
            "model_nss": nss(prob_map, fixation_map),
            "model_cc": cc(prob_map, fixation_density),
            "model_auc": auc_judd(prob_map, fixation_map),
        })



df = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)

if df.empty:
    raise RuntimeError("No prediction files were found. Check PRED_ROOT path.")

metric_columns = [
    "model_entropy",
    "model_center_distance",
    "model_num_peaks",
    "model_nss",
    "model_cc",
    "model_auc",
]

per_image_output = Path("outputs") / f"{MODEL_NAME}_per_image_behaviour.csv"
df.to_csv(per_image_output, index=False)

cat_df = df.groupby("category")[metric_columns].mean().reset_index()

per_category_output = Path("outputs") / f"{MODEL_NAME}_per_category_behaviour.csv"
cat_df.to_csv(per_category_output, index=False)

overall_summary = df[metric_columns].mean().to_dict()
overall_summary.update({
    "model": MODEL_NAME,
    "num_images": int(len(df)),
    "num_categories": int(df["category"].nunique()),
})
overall_df = pd.DataFrame([overall_summary], columns=[
    "model",
    "num_images",
    "num_categories",
    *metric_columns,
])

overall_output = Path("outputs") / f"{MODEL_NAME}_overall_behaviour.csv"
overall_df.to_csv(overall_output, index=False)

print("Saved:")
print(f" - {per_image_output}")
print(f" - {per_category_output}")
print(f" - {overall_output}")
print(f"Processed images: {len(df)}")
print(cat_df)
print(overall_df)