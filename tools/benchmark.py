"""Quality benchmark: score brick grids against the photo the way a viewer sees them.

  python tools/benchmark.py            # saint photo: ours (current settings) vs Brick.me screenshot

Scores (lower dE = better, higher SSIM = better):
  view dE    colour error after blurring both images as the eye does at viewing distance
             (S-CIELAB-style: lightness blurred less than colour) - the headline number
  brick dE   colour error brick by brick (punishes useful dithering; secondary)
  SSIM       structure of the lightness image (edges, features)
  face ...   the same, inside the detected face only
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.colour import deltaE_ciede2000, rgb2lab, ssim  # noqa: E402

# Viewing-distance blur in bricks: about 1-1.5 m from a board with 8 mm studs.
SIGMA_L, SIGMA_AB = 0.7, 1.4


def lab(rgb_u8):
    return rgb2lab(np.asarray(rgb_u8, np.float64) / 255.0)


def blur(lab_img, sl=SIGMA_L, sab=SIGMA_AB):
    out = lab_img.copy()
    out[..., 0] = cv2.GaussianBlur(lab_img[..., 0], (0, 0), sl, borderType=cv2.BORDER_REFLECT)
    for c in (1, 2):
        out[..., c] = cv2.GaussianBlur(lab_img[..., c], (0, 0), sab, borderType=cv2.BORDER_REFLECT)
    return out


def score(grid_rgb, ref_rgb, face=None, valid=None):
    """grid_rgb, ref_rgb: (H, W, 3) uint8 at brick resolution. face/valid: (H, W) bool."""
    h, w = grid_rgb.shape[:2]
    valid = np.ones((h, w), bool) if valid is None else valid
    gm, gr = lab(grid_rgb), lab(ref_rgb)
    de_brick = deltaE_ciede2000(gm, gr)
    de_view = deltaE_ciede2000(blur(gm), blur(gr))
    out = {
        "view dE": float(de_view[valid].mean()),
        "brick dE": float(de_brick[valid].mean()),
        "SSIM": ssim(gr[..., 0], gm[..., 0], data_range=100) * 100,
        # structure as seen from viewing distance: grain is averaged away, edges must stay
        "view SSIM": ssim(blur(gr)[..., 0], blur(gm)[..., 0], data_range=100) * 100,
        # grain: lightness detail in the mosaic that the photo does not have (lower = cleaner)
        "grain": float(np.abs(cv2.Laplacian(gm[..., 0], cv2.CV_64F) - cv2.Laplacian(gr[..., 0], cv2.CV_64F))[valid].mean()),
    }
    if face is not None and (face & valid).any():
        f = face & valid
        out["face view dE"] = float(de_view[f].mean())
        out["face brick dE"] = float(de_brick[f].mean())
        ys, xs = np.where(f)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out["face SSIM"] = ssim(gr[y0:y1, x0:x1, 0], gm[y0:y1, x0:x1, 0], data_range=100, win_size=7) * 100
        bgr, bgm = blur(gr), blur(gm)
        out["face view SSIM"] = ssim(bgr[y0:y1, x0:x1, 0], bgm[y0:y1, x0:x1, 0], data_range=100, win_size=7) * 100
    out["colours"] = len(np.unique(grid_rgb[valid].reshape(-1, 3), axis=0))
    return out


def extract_grid(screenshot, box, cols, rows):
    """Median colour of each brick in a screenshot of a rendered mosaic. box = (x0, y0, x1, y1),
    the board's inner area. Medians skip the stud highlight/shadow and seam lines."""
    im = np.asarray(Image.open(screenshot).convert("RGB"))
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) / cols, (y1 - y0) / rows
    out = np.zeros((rows, cols, 3), np.uint8)
    for r in range(rows):
        for c in range(cols):
            a = im[int(round(y0 + r * py)) + 1:int(round(y0 + (r + 1) * py)) - 1,
                   int(round(x0 + c * px)) + 1:int(round(x0 + (c + 1) * px)) - 1]
            out[r, c] = np.median(a.reshape(-1, 3), axis=0)
    return out


def table(rows):
    keys = ["view dE", "view SSIM", "SSIM", "grain", "face view dE", "face SSIM", "face view SSIM", "colours"]
    print(f"{'':36s}" + "".join(f"{k:>11s}" for k in keys))
    for name, r in rows:
        print(f"{name:36s}" + "".join(f"{r.get(k, float('nan')):11.2f}" if k != "colours" else f"{r.get(k, 0):11d}" for k in keys))
