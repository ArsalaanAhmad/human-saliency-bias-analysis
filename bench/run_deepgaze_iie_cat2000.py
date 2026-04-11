from pathlib import Path
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.special import logsumexp
import deepgaze_pytorch

MANIFEST = Path("bench/manifests/cat2000_balanced_50.csv")
OUT_ROOT = Path("D:/outputs/deepgaze_iie_cat2000_50")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
model.eval()
print("Loaded DeepGazeIIE")

cb_path = Path("bench/centerbias_1024.npy")
cb_template = np.load(cb_path)

def load_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img

def centerbias_for_size(h, w):
    cb = zoom(
        cb_template,
        (h / cb_template.shape[0], w / cb_template.shape[1]),
        order=0,
        mode="nearest"
    )
    cb -= logsumexp(cb)
    return cb.astype(np.float32)

@torch.no_grad()
def predict_log_density(bgr):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    image_tensor = torch.tensor(rgb.transpose(2, 0, 1))[None].to(DEVICE)
    cb = centerbias_for_size(h, w)
    cb_tensor = torch.tensor(cb)[None].to(DEVICE)

    log_density = model(image_tensor, cb_tensor)  # (1,1,H,W)
    return log_density[0, 0].detach().cpu().numpy().astype(np.float32)

def save_outputs(out_base: Path, log_density: np.ndarray):
    out_base.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(out_base.with_suffix(".npy")), log_density)

    vis = np.exp(log_density - log_density.max())
    vis = vis / (vis.max() + 1e-8)
    cv2.imwrite(str(out_base.with_suffix(".png")), (vis * 255).astype(np.uint8))

def main():
    df = pd.read_csv(MANIFEST)
    print(f"Running on {len(df)} images")

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="DeepGazeIIE balanced-50"):
        cat = row.category
        img_name = row.image_name
        img_path = Path(row.image_path)

        out_base = OUT_ROOT / cat / img_path.stem

        if out_base.with_suffix(".npy").exists():
            continue

        bgr = load_bgr(img_path)
        log_density = predict_log_density(bgr)
        save_outputs(out_base, log_density)

    print("Done. Saved outputs to:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()