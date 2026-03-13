from pathlib import Path
import numpy as np
import cv2
import torch
from scipy.ndimage import zoom
from scipy.special import logsumexp
import deepgaze_pytorch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load model
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
model.eval()
print("Loaded DeepGazeIIE")

# Pick one CAT2000 image (first jpg found)
img_path = next(Path("data/Stimuli").rglob("*.jpg"))
print("Image:", img_path)

bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
assert bgr is not None, f"Could not read {img_path}"

# Ensure 3-channel
if bgr.ndim == 2:
    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
if bgr.shape[2] == 4:
    bgr = bgr[:, :, :3]

h, w = bgr.shape[:2]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

# Centerbias (we'll generate one if missing)
cb_path = Path("bench/centerbias_1024.npy")
cb_path.parent.mkdir(parents=True, exist_ok=True)

if not cb_path.exists():
    # make a default centerbias template (1024x1024)
    ys = np.linspace(-1, 1, 1024)
    xs = np.linspace(-1, 1, 1024)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r2 = xx**2 + yy**2
    log_bias = -r2 / (2 * 0.25**2)
    log_bias -= logsumexp(log_bias)
    np.save(cb_path, log_bias.astype(np.float32))
    print("Created", cb_path)

cb_template = np.load(cb_path)

# Resize to image size
cb = zoom(cb_template, (h / cb_template.shape[0], w / cb_template.shape[1]), order=0, mode="nearest")
cb -= logsumexp(cb)

# Tensors
image_tensor = torch.tensor(rgb.transpose(2, 0, 1))[None].to(DEVICE)
cb_tensor = torch.tensor(cb)[None].to(DEVICE)

with torch.no_grad():
    log_density = model(image_tensor, cb_tensor)  # (1,1,H,W)

log_density = log_density[0, 0].detach().cpu().numpy().astype(np.float32)
print("Output:", log_density.shape, "min/max:", float(log_density.min()), float(log_density.max()))

# Save outputs
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

np.save(out_dir / "deepgaze_iie_logdensity.npy", log_density)

vis = np.exp(log_density - log_density.max())
vis = vis / (vis.max() + 1e-8)
cv2.imwrite(str(out_dir / "deepgaze_iie_vis.png"), (vis * 255).astype(np.uint8))

print("Saved outputs/deepgaze_iie_vis.png")