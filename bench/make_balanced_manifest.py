from pathlib import Path
import csv

STIM_ROOT = Path("data/Stimuli")
OUT_CSV = Path("bench/manifests/cat2000_balanced_50.csv")
N_PER_CATEGORY = 35

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp"}
rows = []

for cat_dir in sorted([p for p in STIM_ROOT.iterdir() if p.is_dir()]):
    images = sorted([p for p in cat_dir.iterdir() if p.suffix.lower() in exts])[:N_PER_CATEGORY]

    for img_path in images:
        rows.append({
            "category": cat_dir.name,
            "image_name": img_path.name,
            "image_path": str(img_path)
        })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["category", "image_name", "image_path"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved manifest to {OUT_CSV}")
print(f"Total rows: {len(rows)}")

