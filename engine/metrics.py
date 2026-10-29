"""How close a mosaic is to the photo.

The reference is the untouched crop averaged over each brick's area: the best any N x N
grid of solid colours can show. Scores compare the mosaic's brick colours to it.
"""
import cv2
import numpy as np
from skimage.color import deltaE_ciede2000
from skimage.metrics import structural_similarity

from .palette import hex_to_rgb, rgb_to_lab

JND = 2.3  # just-noticeable colour difference (CIEDE2000)


def reference_cells(crop_rgb, size):
    return cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_AREA)


def similarity(matrix, crop_rgb, face_mask=None):
    n = matrix["size"]
    hexes = {e["code"]: hex_to_rgb(e["hex"]) for e in matrix["palette"]["colors"]}
    mosaic = np.array([[hexes[c] for c in row] for row in matrix["grid"]], np.uint8)
    ref = reference_cells(crop_rgb, n)
    lm, lr = rgb_to_lab(mosaic), rgb_to_lab(ref)
    de = deltaE_ciede2000(lm.reshape(-1, 3), lr.reshape(-1, 3)).reshape(n, n)
    ssim = structural_similarity(lr[..., 0], lm[..., 0], data_range=100, win_size=7)
    out = {
        "mean_delta_e": round(float(de.mean()), 2),
        "pct_unnoticeable": round(float((de <= JND).mean() * 100), 1),
        "pct_close": round(float((de <= 5).mean() * 100), 1),
        "structure_ssim": round(float(ssim * 100), 1),
    }
    # Headline score: share of bricks whose colour is a close match (CIEDE2000 <= 5,
    # a common "commercial match" tolerance), weighted with structure so blur is penalised.
    out["similarity"] = round(0.5 * out["pct_close"] + 0.5 * out["structure_ssim"], 1)
    if face_mask is not None:
        fm = cv2.resize(face_mask.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA) > 0.5
        if fm.any():
            out["face_mean_delta_e"] = round(float(de[fm].mean()), 2)
    return out
