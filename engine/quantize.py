"""Steps 7-8: area downscale to the board (W x H bricks), CIEDE2000 matching, optional LAB dithering."""
import cv2
import numpy as np

from .colour import deltaE_ciede2000
from .palette import ciede2000_matrix, lab_to_rgb_u8, rgb_to_lab


def _wh(size):
    return (size, size) if isinstance(size, (int, np.integer)) else tuple(size)


def downscale(rgb, size):
    """Each brick is the average of its area (INTER_AREA), never nearest-neighbour.
    size = N or (width, height)."""
    return cv2.resize(rgb, _wh(size), interpolation=cv2.INTER_AREA)


def downscale_mask(mask, size):
    return cv2.resize(mask.astype(np.float32), _wh(size), interpolation=cv2.INTER_AREA)


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


FILTERS = ("none", "bw", "sepia", "warm", "cool", "vivid")


def adjust(cells_rgb, s):
    """User adjustments at brick scale, in LAB: filter, brightness, contrast, saturation,
    highlights and detail (lightness unsharp over the whole picture). All neutral at defaults.

    highlights > 0 stretches only the tones above mid-grey (L* 45): shadows, midtones and blacks
    stay as they are, lit skin and pale areas reach the light palette colours (peach, beige)
    instead of mid browns. highlights 40 maps the photo's L* 65 to about 73."""
    filt = s.get("filter", "none")
    bright, contrast = s.get("brightness", 0) / 100, s.get("contrast", 0) / 100
    sat, detail = s.get("saturation_adj", 0) / 100, s.get("detail", 0)
    highlights = s.get("highlights", 0) / 100
    if filt == "none" and not (bright or contrast or sat or detail or highlights):
        return cells_rgb
    lab = rgb_to_lab(cells_rgb)
    L, ab = lab[..., 0], lab[..., 1:]
    if filt == "bw":
        ab[:] = 0
    elif filt == "sepia":
        ab[:] = [6.0, 16.0]
        ab *= (0.35 + 0.65 * np.clip(L / 100, 0, 1))[..., None]
    elif filt == "warm":
        ab += [3.0, 8.0]
    elif filt == "cool":
        ab += [-2.0, -9.0]
    elif filt == "vivid":
        # Rich look measured from the reference product (Brick.me) on a painted portrait: a little
        # more colour, lifted highlights, crisper features. Paired with the over-saturation guard
        # in the pipeline so lifted pastels do not turn into vivid oranges.
        sat += 0.10
        highlights += 0.20
        detail += 1.0
    L += bright * 25
    if highlights:
        pivot = 45.0
        L[:] = np.where(L > pivot, pivot + (L - pivot) * (1 + highlights), L)
    L[:] = 50 + (L - 50) * (1 + contrast)
    ab *= 1 + sat
    if detail:
        L += (L - cv2.GaussianBlur(L, (0, 0), 0.9)) * detail * 0.35
    np.clip(L, 0, 100, out=L)
    return lab_to_rgb_u8(lab)


def match(cells_rgb, palette, kL=1.0, kC=1.0, oversat=0.0):
    """Nearest palette index per cell by CIEDE2000. kL < 1 weights lightness more, kC < 1 chroma;
    oversat penalises colours much more saturated than the target (see ciede2000_matrix)."""
    h, w = cells_rgb.shape[:2]
    lab = rgb_to_lab(cells_rgb).reshape(-1, 3)
    return ciede2000_matrix(lab, palette.lab, kL, kC=kC, oversat=oversat).argmin(axis=1).reshape(h, w)


def dither(cells_rgb, palette, fmask, strength=0.8, face_strength=0.3, clamp=12.0, kL=1.0):
    """Serpentine Floyd-Steinberg in LAB with clamped error, weaker inside faces."""
    h, w = cells_rgb.shape[:2]
    work = rgb_to_lab(cells_rgb).astype(np.float64)
    gain = strength + (face_strength - strength) * np.clip(fmask, 0, 1)
    pal = palette.lab
    out = np.zeros((h, w), np.int32)
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in xs:
            px = work[y, x]
            i = int(deltaE_ciede2000(np.broadcast_to(px, pal.shape), pal, kL=kL).argmin())
            out[y, x] = i
            err = np.clip((px - pal[i]) * gain[y, x], -clamp, clamp)
            if 0 <= x + d < w:
                work[y, x + d] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    work[y + 1, x - d] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= x + d < w:
                    work[y + 1, x + d] += err * 1 / 16
    return out


def _gauss(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT) if sigma > 0 else img


def optimise_view(cells_rgb, palette, idx, sigma_l=0.7, sigma_ab=1.4, fidelity=0.3, kL=1.0,
                  passes=8, wmap=None, max_extra=None, kC=1.0, oversat=0.0):
    """Improve a brick assignment for how the mosaic looks from viewing distance.

    Minimises  sum_c w_c ||G_c * (M_c - T_c)||^2  +  fidelity * sum w ||M - T||^2  over the
    palette indices, where T is the target (cells) and M the mosaic in CIELAB, G a Gaussian per
    channel (lightness blurred less than colour, as the eye does), w_L = 1 / kL^2. This is
    Direct Binary Search halftoning on the brick grid: a brick changes colour only when that
    makes the blurred picture closer to the photo, so gradients and edges come out better than
    brick-by-brick nearest matching, without random noise. Deterministic.

    Updates bricks on a lattice spaced wider than the kernels' effective reach, so changes made
    together do not interfere and every pass lowers the objective. fidelity may be a number or an
    (H, W) map: high values keep bricks close to their own target (clean, e.g. on faces), low
    values let the eye mix neighbouring colours (smooth gradients, e.g. on backgrounds).
    wmap (H, W) optionally weights regions in the viewing term. max_extra limits each brick to
    colours at most this much further (CIEDE2000) from its target than its nearest colour, so
    blending only mixes close shades - never an off-hue colour that is right only on average."""
    T = rgb_to_lab(cells_rgb).astype(np.float64)
    P = palette.lab.astype(np.float64)
    h, w = idx.shape
    wc = np.array([1.0 / kL ** 2, 1.0, 1.0])
    sig = (sigma_l, sigma_ab, sigma_ab)
    W = np.ones((h, w)) if wmap is None else np.asarray(wmap, np.float64)
    F = np.broadcast_to(np.asarray(fidelity, np.float64), (h, w))
    # Fidelity uses the same CIEDE2000 distance as the initial matching, so a brick's nearest
    # colour always has the lowest fidelity cost (a different metric here would silently swap
    # bricks to colours that are only "closer" by that other metric).
    D = ciede2000_matrix(T.reshape(-1, 3), P, kL, kC=kC, oversat=oversat).reshape(h, w, len(P))
    D2 = D ** 2
    banned = None
    if max_extra is not None:
        banned = D > D.min(-1, keepdims=True) + max_extra
    idx = idx.copy()

    def err_image(M):
        return [(_gauss(M[..., c] - T[..., c], sig[c])) for c in range(3)]

    # energy of each kernel (sum of K^2): the cost of a unit change at one brick
    k2 = []
    for c in range(3):
        imp = np.zeros((2 * 8 * 3 + 1,) * 2)
        imp[imp.shape[0] // 2, imp.shape[1] // 2] = 1.0
        k = _gauss(imp, sig[c])
        k2.append(float((k ** 2).sum()))
    reach = int(np.ceil(2 * 3 * max(sig)))  # K*K support radius (3 sigma each)
    step = reach + 1

    M = P[idx]
    E = err_image(M)
    for _ in range(passes):
        changed = 0
        for oy in range(step):
            for ox in range(step):
                G = [_gauss(E[c] * W, sig[c]) for c in range(3)]
                ys = np.arange(oy, h, step)
                xs = np.arange(ox, w, step)
                if not len(ys) or not len(xs):
                    continue
                yy, xx = np.meshgrid(ys, xs, indexing="ij")
                yy, xx = yy.ravel(), xx.ravel()
                cur = M[yy, xx]                              # (n, 3)
                d = P[None, :, :] - cur[:, None, :]          # (n, P, 3)
                g = np.stack([G[c][yy, xx] for c in range(3)], -1)  # (n, 3)
                wq = W[yy, xx][:, None]
                dview = (wc * (2 * d * g[:, None, :] + d ** 2 * np.array(k2) * wq[..., None])).sum(-1)
                d2 = D2[yy, xx]                              # (n, P)
                dfid = (d2 - d2[np.arange(len(yy)), idx[yy, xx]][:, None]) * wq
                dj = dview + F[yy, xx][:, None] * dfid
                if banned is not None:
                    dj = np.where(banned[yy, xx], np.inf, dj)
                best = dj.argmin(1)
                gain = dj[np.arange(len(best)), best]
                ok = gain < -1e-9
                if ok.any():
                    yy, xx, best = yy[ok], xx[ok], best[ok]
                    idx[yy, xx] = best
                    M[yy, xx] = P[best]
                    changed += int(ok.sum())
                    E = err_image(M)
        if not changed:
            break
    return idx


def feature_lines(cells_rgb, fmask, amount=1.5, threshold=4.0, size=3):
    """Ink-like feature lines: deepen thin dark valleys (eyelids, brows, nostrils, lip line)
    inside faces, so they become solid dark bricks instead of soft browns.

    A morphological black-hat (closing minus image) on L* at brick scale finds dark features
    up to `size` bricks wide; valleys deeper than `threshold` L* are darkened by `amount` times
    their depth. Colour (a*, b*) is kept; only lightness changes, only where fmask > 0."""
    if amount <= 0 or fmask is None or not np.any(fmask > 0.05):
        return cells_rgb
    lab = rgb_to_lab(cells_rgb)
    L = lab[..., 0].astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    depth = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k) - L
    depth = np.maximum(depth - threshold, 0)
    weight = np.clip(fmask, 0, 1)
    lab[..., 0] = np.clip(L - amount * depth * weight, 0, 100)
    return lab_to_rgb_u8(lab)
