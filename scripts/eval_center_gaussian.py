import cv2
import numpy as np
from scipy.io import loadmat
from collections import defaultdict
from cat2000_index import build_pairs
from metrics import nss

def center_gaussian(H, W):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    sigma = min(H, W) / 6
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)

pairs = build_pairs()
scores = []
by_cat = defaultdict(list)

for cat, img_path, fix_path in pairs:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    H, W, _ = img.shape

    fix = loadmat(fix_path)["fixLocs"].astype(np.float32)
    fix = cv2.resize(fix, (W, H), interpolation=cv2.INTER_NEAREST)

    pred = center_gaussian(H, W)

    s = float(nss(pred, fix))
    scores.append(s)
    by_cat[cat].append(s)

scores_np = np.array(scores)
print("CENTER GAUSSIAN NSS (all): mean =", scores_np.mean(), "std =", scores_np.std())

print("\nPer-category means (sorted low->high):")
cat_means = sorted([(c, np.mean(v), np.std(v), len(v)) for c, v in by_cat.items()], key=lambda x: x[1])
for c, m, sd, n in cat_means:
    print(f"{c:15s} mean={m:.3f} std={sd:.3f} n={n}")