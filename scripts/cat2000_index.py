import os
import glob

STIM_ROOT = r"data/Stimuli"
FIX_ROOT  = r"data/FIXATIONLOCS"

def list_categories():
    return sorted([d for d in os.listdir(STIM_ROOT) if os.path.isdir(os.path.join(STIM_ROOT, d))])

def build_pairs():
    pairs = []
    for cat in list_categories():
        img_dir = os.path.join(STIM_ROOT, cat)
        fix_dir = os.path.join(FIX_ROOT, cat)

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if len(img_paths) == 0:
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))  # fallback

        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            fp = os.path.join(fix_dir, stem + ".mat")
            if os.path.exists(fp):
                pairs.append((cat, ip, fp))
    return pairs

if __name__ == "__main__":
    pairs = build_pairs()
    print("Total pairs:", len(pairs))
    print("Example:", pairs[0])