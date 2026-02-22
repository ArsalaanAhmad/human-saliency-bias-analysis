import os
import numpy as np
import cv2
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import entropy

STIM_ROOT = "data/Stimuli"
FIX_ROOT = "data/FIXATIONLOCS"

def list_categories():
    return sorted([d for d in os.listdir(STIM_ROOT) if os.path.isdir(os.path.join(STIM_ROOT, d))])

def compute_entropy(fix_map):
    prob = fix_map.flatten().astype(np.float32)
    if prob.sum() == 0:
        return 0.0
    prob /= prob.sum()
    return entropy(prob + 1e-8)

results = []

categories = list_categories()

for cat in categories:
    stim_dir = os.path.join(STIM_ROOT, cat)
    fix_dir = os.path.join(FIX_ROOT, cat)

    distances = []
    variances = []
    entropies = []

    for fname in tqdm(os.listdir(fix_dir), desc=f"Processing {cat}"):
        if not fname.endswith(".mat"):
            continue

        fix_path = os.path.join(fix_dir, fname)
        fix = loadmat(fix_path)["fixLocs"].astype(np.float32)

        H, W = fix.shape
        cy, cx = H / 2, W / 2

        ys, xs = np.where(fix > 0)

        if len(xs) == 0:
            continue

        # Center distance
        dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        distances.append(np.mean(dists))

        # Dispersion
        variances.append(np.var(xs) + np.var(ys))

        # Entropy
        entropies.append(compute_entropy(fix))

    results.append({
        "category": cat,
        "mean_center_distance": np.mean(distances),
        "mean_dispersion": np.mean(variances),
        "mean_entropy": np.mean(entropies)
    })

df = pd.DataFrame(results)
df = df.sort_values("mean_center_distance")

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/human_stats_by_category.csv", index=False)

print("\nSaved to outputs/human_stats_by_category.csv")
print(df)
