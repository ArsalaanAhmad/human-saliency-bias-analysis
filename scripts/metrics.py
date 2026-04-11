import numpy as np
from scipy.ndimage import gaussian_filter


EPS = 1e-8


def normalize_map(saliency_map):
    saliency_map = saliency_map.astype(np.float32)
    mean = saliency_map.mean()
    std = saliency_map.std()
    return (saliency_map - mean) / (std + EPS)


def density_from_fixation_map(fixation_map, sigma=15):
    density = gaussian_filter(fixation_map.astype(np.float32), sigma=sigma)
    max_value = float(density.max())
    if max_value > 0:
        density /= max_value
    return density


def nss(saliency_map, fixation_map):
    saliency_norm = normalize_map(saliency_map)
    fixation_mask = fixation_map > 0
    if not np.any(fixation_mask):
        return float("nan")
    return float(saliency_norm[fixation_mask].mean())


def cc(saliency_map, target_map):
    saliency_norm = normalize_map(saliency_map)
    target_norm = normalize_map(target_map)
    numerator = float(np.mean(saliency_norm * target_norm))
    denominator = float(saliency_norm.std() * target_norm.std())
    if denominator <= EPS:
        return float("nan")
    return numerator / denominator


def auc_judd(saliency_map, fixation_map):
    saliency_map = saliency_map.astype(np.float32)
    fixation_mask = fixation_map > 0

    if not np.any(fixation_mask):
        return float("nan")

    saliency_min = float(saliency_map.min())
    saliency_max = float(saliency_map.max())
    if saliency_max - saliency_min <= EPS:
        return float("nan")

    saliency_norm = (saliency_map - saliency_min) / (saliency_max - saliency_min + EPS)
    positives = saliency_norm[fixation_mask]
    negatives = saliency_norm[~fixation_mask]

    if negatives.size == 0:
        return float("nan")

    thresholds = np.sort(positives)[::-1]
    tp = np.zeros(thresholds.size + 2, dtype=np.float32)
    fp = np.zeros(thresholds.size + 2, dtype=np.float32)
    tp[-1] = 1.0
    fp[-1] = 1.0

    num_positives = float(positives.size)
    num_negatives = float(negatives.size)

    for index, threshold in enumerate(thresholds, start=1):
        tp[index] = index / num_positives
        fp[index] = np.count_nonzero(negatives >= threshold) / num_negatives

    return float(np.trapz(tp, fp))