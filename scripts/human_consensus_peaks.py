import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, maximum_filter
from tqdm import tqdm

STIM_ROOT = "data/Stimuli"
FIX_ROOT = "data/FIXATIONLOCS"

def list_categories():
    return sorted([d for d in os.listdir(FIX_ROOT) if os.path.isdir(os.path.join(FIX_ROOT, d))])

def load_fix(path):
    return loadmat(path)["fixLocs"].astype(np.float32)

def density_from_fix(fix, sigma=15):
    d = gaussian_filter(fix, sigma=sigma)
    if d.max() > 0:
        d /= d.max()
    return d

def peakiness(density):
    # Convert to probability distribution
    p = density.astype(np.float32)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    return float(p.max())

def count_peaks(density, threshold=0.6, neighborhood=21):
    """
    Count local maxima above threshold.
    density assumed normalized to [0,1]
    """
    # local maxima
    local_max = (density == maximum_filter(density, size=neighborhood))
    peaks = local_max & (density >= threshold)
    return int(peaks.sum())

rows = []

for cat in list_categories():
    fix_dir = os.path.join(FIX_ROOT, cat)

    peak_vals = []
    n_peaks_vals = []
    fix_counts = []

    for fname in tqdm(os.listdir(fix_dir), desc=f"{cat}"):
        if not fname.endswith(".mat"):
            continue
        fix = load_fix(os.path.join(fix_dir, fname))
        fix_counts.append(float(fix.sum()))

        dens = density_from_fix(fix, sigma=15)  # tweak sigma later if needed
        peak_vals.append(peakiness(dens))
        n_peaks_vals.append(count_peaks(dens, threshold=0.6, neighborhood=31))

    rows.append({
        "category": cat,
        "mean_fix_count": np.mean(fix_counts),
        "mean_peak_prob": np.mean(peak_vals),
        "mean_num_peaks": np.mean(n_peaks_vals),
        "std_num_peaks": np.std(n_peaks_vals)
    })

df = pd.DataFrame(rows).sort_values("mean_peak_prob", ascending=False)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/human_consensus_peaks_by_category.csv", index=False)

print(df)
print("\nSaved outputs/human_consensus_peaks_by_category.csv")