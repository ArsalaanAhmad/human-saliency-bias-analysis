import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from scipy.ndimage import maximum_filter

PRED_ROOT = Path("outputs/deepgaze_iie_cat2000")

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

rows = []

for cat_dir in sorted([p for p in PRED_ROOT.iterdir() if p.is_dir()]):
    for npy_path in tqdm(sorted(cat_dir.glob("*.npy")), desc=f"Analysing {cat_dir.name}"):
        log_density = np.load(npy_path)
        prob_map = np.exp(log_density - log_density.max())
        prob_map = prob_map / (prob_map.sum() + 1e-8)

        rows.append({
            "category": cat_dir.name,
            "image": npy_path.stem,
            "model_entropy": compute_entropy(prob_map),
            "model_center_distance": center_distance(prob_map),
            "model_num_peaks": count_peaks(prob_map)
        })

df = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/deepgaze_iie_per_image_behaviour.csv", index=False)

cat_df = df.groupby("category").agg({
    "model_entropy": "mean",
    "model_center_distance": "mean",
    "model_num_peaks": "mean"
}).reset_index()

cat_df.to_csv("outputs/deepgaze_iie_per_category_behaviour.csv", index=False)

print("Saved:")
print(" - outputs/deepgaze_iie_per_image_behaviour.csv")
print(" - outputs/deepgaze_iie_per_category_behaviour.csv")
print(cat_df)