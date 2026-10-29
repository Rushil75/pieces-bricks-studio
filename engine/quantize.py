"""Steps 7-8: area downscale to N x N, CIEDE2000 matching, optional LAB dithering."""
import cv2
import numpy as np

from .colour import deltaE_ciede2000
from .palette import ciede2000_matrix, lab_to_rgb_u8, rgb_to_lab


def downscale(rgb, size):
    """Each brick is the average of its area (INTER_AREA), never nearest-neighbour."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def downscale_mask(mask, size):
    return cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def face_detail(cells_rgb, fmask, amount=0.5, sigma=0.8):
    """Unsharp mask on L* at brick scale, inside faces only, so eyes/brows/mouth stay
    distinct after area averaging. Colour (a*, b*) is untouched, so skin stays natural."""
    if amount <= 0:
        return cells_rgb
    lab = rgb_to_lab(cells_rgb)
    L = lab[..., 0]
    blur = cv2.GaussianBlur(L, (0, 0), sigma)
    lab[..., 0] = np.clip(L + (L - blur) * amount * np.clip(fmask, 0, 1), 0, 100)
    return lab_to_rgb_u8(lab)


def match(cells_rgb, palette, kL=1.0):
    """Nearest palette index per cell by CIEDE2000. kL < 1 weights lightness more."""
    n = cells_rgb.shape[0]
    lab = rgb_to_lab(cells_rgb).reshape(-1, 3)
    return ciede2000_matrix(lab, palette.lab, kL).argmin(axis=1).reshape(n, n)


def dither(cells_rgb, palette, fmask, strength=0.8, face_strength=0.3, clamp=12.0, kL=1.0):
    """Serpentine Floyd-Steinberg in LAB with clamped error, weaker inside faces."""
    n = cells_rgb.shape[0]
    work = rgb_to_lab(cells_rgb).astype(np.float64)
    gain = strength + (face_strength - strength) * np.clip(fmask, 0, 1)
    pal = palette.lab
    out = np.zeros((n, n), np.int32)
    for y in range(n):
        xs = range(n) if y % 2 == 0 else range(n - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in xs:
            px = work[y, x]
            i = int(deltaE_ciede2000(np.broadcast_to(px, pal.shape), pal, kL=kL).argmin())
            out[y, x] = i
            err = np.clip((px - pal[i]) * gain[y, x], -clamp, clamp)
            if 0 <= x + d < n:
                work[y, x + d] += err * 7 / 16
            if y + 1 < n:
                if 0 <= x - d < n:
                    work[y + 1, x - d] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= x + d < n:
                    work[y + 1, x + d] += err * 1 / 16
    return out
