# Template for running a saliency model on the CAT2000 dataset and saving predictions.
# Usage: python bench/run_model_template.py

import os

import pandas as pd
import cv2
from tqdm import tqdm

MANIFEST = "bench/manifests/cat2000_all.csv"

def predict_one(bgr_image):
    """
    Return a float32 saliency map (H,W) in [0,1] or arbitrary scale.
    Uniqye to model. This is where i run saliency model on the input image.
    """
    raise NotImplementedError

def main(model_name):
    df = pd.read_csv(MANIFEST)
    for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"Run {model_name}"):
        cat, img_name, img_path = r.category, r.image_name, r.image_path
        stem = os.path.splitext(img_name)[0]

        out_dir = os.path.join("bench","preds",model_name,cat)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}.png")
        if os.path.exists(out_path):
            continue

        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is None:
            continue

        sal = predict_one(im)  # (H,W) float
        sal = sal.astype("float32")
        # normalize to 0..255 for saving
        sal = sal - sal.min()
        if sal.max() > 0: sal = sal / sal.max()
        sal_u8 = (sal * 255.0).astype("uint8")
        cv2.imwrite(out_path, sal_u8)

if __name__ == "__main__":
    main("MODELNAME")