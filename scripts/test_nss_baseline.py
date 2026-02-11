import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from metrics import nss

IMG_PATH = r"data/Stimuli/Action/001.jpg"
FIX_PATH = r"data/FIXATIONLOCS/Action/001.mat"

# Load image
img = cv2.imread(IMG_PATH)
H, W, _ = img.shape

# Load fixation map
fix = loadmat(FIX_PATH)["fixLocs"].astype(np.float32)
fix = cv2.resize(fix, (W, H), interpolation=cv2.INTER_NEAREST)

# Build center Gaussian baseline
yy, xx = np.mgrid[0:H, 0:W]
center_y, center_x = H / 2, W / 2
sigma = min(H, W) / 6
center_gaussian = np.exp(-((xx-center_x)**2 + (yy-center_y)**2)/(2*sigma**2))

score = nss(center_gaussian, fix)

print("Center Gaussian NSS:", score)
