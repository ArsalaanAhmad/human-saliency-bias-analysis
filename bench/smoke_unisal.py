import os, sys
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party", "unisal"))

from unisal.model import UNISAL

MANIFEST = os.path.join(ROOT, "bench", "manifests", "cat2000_all.csv")
WEIGHTS  = os.path.join(ROOT, "third_party", "unisal", "training_runs", "pretrained_unisal", "weights_ft_mit1003.pth")
OUT_ROOT = os.path.join(ROOT, "bench", "preds", "unisal")

def preprocess(img_any):
    """
    Accepts grayscale or color numpy image and returns (1,3,H,W) float tensor.
    """
    import numpy as np
    import cv2
    import torch

    if img_any is None:
        raise ValueError("preprocess got None")

    # If already grayscale (H,W), convert to 3-channel
    if img_any.ndim == 2:
        img_any = cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)

    # If has alpha (H,W,4), drop alpha
    if img_any.ndim == 3 and img_any.shape[2] == 4:
        img_any = img_any[:, :, :3]

    # If something still wrong, crash loudly
    if not (img_any.ndim == 3 and img_any.shape[2] == 3):
        raise ValueError(f"Expected image with 3 channels, got shape {img_any.shape}")

    rgb = cv2.cvtColor(img_any, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # FINAL GUARANTEE: if somehow 1-channel slips through, repeat it
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    return x

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNISAL(
        sources=("MIT1003",),
        ds_bn=False,
        ds_adaptation=False,
        ds_smoothing=False,
        ds_gaussians=False,
        verbose=0
    ).to(device).eval()

    ckpt = torch.load(WEIGHTS, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded weights.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    return model, device

@torch.no_grad()
def predict_one(model, device, bgr):
    x = preprocess(bgr).to(device).float().contiguous()  # (1,3,H,W)

    # HARD GUARD: if UNISAL internally tries to treat it as grayscale, we block that by duplicating channels again
    if x.shape[1] != 3:
        x = x.repeat(1, 3, 1, 1)

    # DEBUG: prove right before forward
    print("Tensor shape to model:", tuple(x.shape))

    out = model(x)

    if isinstance(out, dict):
        out = list(out.values())[0]
    if isinstance(out, (list, tuple)):
        out = out[0]

    sal = out.squeeze().detach().cpu().numpy().astype(np.float32)
    return sal

def main(limit=10):
    df = pd.read_csv(MANIFEST).head(limit)
    model, device = load_model()

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="UNISAL smoke"):
        cat, img_name, img_path = r.category, r.image_name, r.image_path
        stem = os.path.splitext(img_name)[0]

        out_dir = os.path.join(OUT_ROOT, cat)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}.png")
        if os.path.exists(out_path):
            continue

        bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            continue

        if len(bgr.shape) == 2:  # grayscale
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

        if bgr.shape[2] == 4:  # Drop alpha if present
            bgr = bgr[:,:,:3]

        if bgr.ndim == 2:
          print("Grayscale image:", img_path)

        sal = predict_one(model, device, bgr)

        sal = sal - sal.min()
        if sal.max() > 0:
            sal = sal / sal.max()
        sal_u8 = (sal * 255).astype(np.uint8)

        cv2.imwrite(out_path, sal_u8)

    print("Done. Wrote predictions to:", OUT_ROOT)

if __name__ == "__main__":
    main(limit=10)