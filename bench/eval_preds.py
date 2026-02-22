# Script that evaluates predicted saliency maps against CAT2000 fixation data using various metrics.
# Usage: python bench/eval_preds.py <model_name>

import os, csv, glob
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import entropy as ent

FIX_ROOT = "data/FIXATIONLOCS"
MANIFEST = "bench/manifests/cat2000_all.csv"

def load_fix(cat, img_name):
    # CAT2000 naming: image "053.jpg" -> "053.mat"
    stem = os.path.splitext(img_name)[0]
    mat_path = os.path.join(FIX_ROOT, cat, f"{stem}.mat")
    m = loadmat(mat_path)
    fix = m["fixLocs"].astype(np.float32)
    return fix

def to_density(x, sigma=15):
    d = gaussian_filter(x.astype(np.float32), sigma=sigma)
    s = d.sum()
    if s > 0: d /= s
    return d

def nss(pred, fix):
    p = pred.astype(np.float32)
    p = (p - p.mean()) / (p.std() + 1e-8)
    pts = fix > 0
    if pts.sum() == 0: return np.nan
    return float(p[pts].mean())

def cc(pred, gt):
    a = pred.flatten().astype(np.float32)
    b = gt.flatten().astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a*b).mean())

def sim(pred, gt):
    p = pred.astype(np.float32); p = p / (p.sum() + 1e-8)
    g = gt.astype(np.float32); g = g / (g.sum() + 1e-8)
    return float(np.minimum(p, g).sum())

def center_distance(density):
    H,W = density.shape
    cy,cx = H/2, W/2
    ys,xs = np.where(density > 0)
    if len(xs)==0: return np.nan
    # weight by density
    w = density[ys,xs]
    w = w / (w.sum()+1e-8)
    d = np.sqrt((xs-cx)**2 + (ys-cy)**2)
    return float((w*d).sum())

def density_entropy(density):
    p = density.flatten()
    p = p / (p.sum()+1e-8)
    return float(ent(p + 1e-12))

def num_peaks(density, thr=0.6, neigh=31):
    # density should be normalized 0..1
    if density.max() > 0:
        d = density / density.max()
    else:
        d = density
    local_max = (d == maximum_filter(d, size=neigh))
    peaks = local_max & (d >= thr)
    return int(peaks.sum())

def eval_model(model_name):
    df = pd.read_csv(MANIFEST)
    rows = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"Eval {model_name}"):
        cat, img_name, img_path = r.category, r.image_name, r.image_path
        stem = os.path.splitext(img_name)[0]
        pred_path = os.path.join("bench","preds",model_name,cat,f"{stem}.png")
        if not os.path.exists(pred_path):
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        pred = pred.astype(np.float32)

        fix = load_fix(cat, img_name)
        gt = to_density(fix, sigma=15)

        pred_d = to_density(pred, sigma=15)

        rows.append({
            "category": cat,
            "image": stem,
            "NSS": nss(pred_d, fix),
            "CC": cc(pred_d, gt),
            "SIM": sim(pred_d, gt),
            "pred_center_dist": center_distance(pred_d),
            "pred_entropy": density_entropy(pred_d),
            "pred_num_peaks": num_peaks(pred_d)
        })

    out_dir = os.path.join("bench","results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_name}_per_image.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # per-category summary
    summ = pd.DataFrame(rows).groupby("category")[["NSS","CC","SIM","pred_center_dist","pred_entropy","pred_num_peaks"]].mean().reset_index()
    summ_csv = os.path.join(out_dir, f"{model_name}_per_category.csv")
    summ.to_csv(summ_csv, index=False)
    print("Saved:", summ_csv)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bench/eval_preds.py <model_name>")
        raise SystemExit(1)
    eval_model(sys.argv[1])