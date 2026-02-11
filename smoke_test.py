import os, glob
import cv2
import matplotlib.pyplot as plt

IMG_DIR = r"data/images"
FIXMAP_DIR = r"data/fixations"

img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
fixmap_paths = sorted(glob.glob(os.path.join(FIXMAP_DIR, "*_fixMap.jpg")))

print("Images:", len(img_paths))
print("FixMaps:", len(fixmap_paths))

assert len(img_paths) > 0
assert len(fixmap_paths) > 0

img_path = img_paths[0]
fixmap_path = fixmap_paths[0]

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
fixmap = cv2.imread(fixmap_path, cv2.IMREAD_GRAYSCALE)

print("Example image:", os.path.basename(img_path), img.shape)
print("Example fixmap:", os.path.basename(fixmap_path), fixmap.shape)

plt.figure()
plt.imshow(img); plt.title("Image"); plt.axis("off")
plt.show()

plt.figure()
plt.imshow(fixmap, cmap="gray"); plt.title("Fixation density map"); plt.axis("off")
plt.show()