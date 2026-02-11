import numpy as np

def normalize_map(saliency):
    saliency = saliency.astype(np.float32)
    mean = saliency.mean()
    std = saliency.std() + 1e-8
    return (saliency - mean) / std

def nss(saliency_map, fixation_map):
    """
    saliency_map: continuous prediction (H,W)
    fixation_map: binary map (H,W)
    """
    saliency_norm = normalize_map(saliency_map)
    return saliency_norm[fixation_map > 0].mean()