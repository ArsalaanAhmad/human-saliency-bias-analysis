import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

IMG_PATH = r"data/Stimuli/Action/001.jpg"
FIX_PATH = r"data/FIXATIONLOCS/Action/001.mat"

# Load image
img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
assert img_bgr is not None, f"Could not read image: {IMG_PATH}"
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W, _ = img.shape

# Load fixation map
mat = loadmat(FIX_PATH)
fix = mat["fixLocs"]  # shape (1080, 1920)
fix = fix.astype(np.float32)

print("Image shape:", img.shape)
print("Fix map shape:", fix.shape, "unique:", np.unique(fix)[:10])

# If image isn't 1080x1920, resize fix map to match image for overlay
if fix.shape != (H, W):
    fix_resized = cv2.resize(fix, (W, H), interpolation=cv2.INTER_NEAREST)
else:
    fix_resized = fix

# Make a density map (Gaussian blur) from binary fix map
density = gaussian_filter(fix_resized, sigma=15)
density = density / (density.max() + 1e-8)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(fix_resized, cmap="gray")
plt.title("Fixation Map (binary)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.imshow(density, alpha=0.5, cmap="jet")
plt.title("Density overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
