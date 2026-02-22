# Script to create a manifest CSV file for the CAT2000 dataset, listing each image along with its category and file path.
# Usage: python bench/make_manifest.py

import os, csv, glob
from scipy.io import loadmat

STIM_ROOT = "data/Stimuli"
OUT = "bench/manifests/cat2000_all.csv"

os.makedirs("bench/manifests", exist_ok=True)

rows = []
for cat in sorted(os.listdir(STIM_ROOT)):
    cdir = os.path.join(STIM_ROOT, cat)
    if not os.path.isdir(cdir): 
        continue
    for img in sorted(glob.glob(os.path.join(cdir, "*"))):
        if img.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            rows.append((cat, os.path.basename(img), img))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category","image_name","image_path"])
    w.writerows(rows)

print("Wrote", OUT, "rows:", len(rows))