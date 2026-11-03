"""Studio engine: a faithful port of the reference brick-mosaic demo (single-page JS app).

Per board size:
  1. crop on the faces (tighter on small boards), resample to 8 px per stud
  2. background simplify outside a head-and-shoulders oval, brighten the face (gamma)
  3. auto levels, contrast, saturation
  4. area-average each stud in linear light, convert to OKLab
  5. edge-preserving smoothing, local contrast, dark lines, light sharpening (on OKLab L)
  6. reduce the palette (drop the least useful colours, and colours used by < 0.8 % of studs)
  7. nearest colour in OKLab with lightness weighted 1.3 (optional Floyd-Steinberg)

Arrays that are Float32Array / Uint8ClampedArray in the original are float32 / uint8 here, and
loops keep the original order, so results match the reference brick for brick on the same input.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .palette import Palette, load_client_palette, CONFIG_DIR
from .preprocess import detect_faces, face_mask, load_image

CELL = 8  # working pixels per stud

_c = np.arange(256) / 255.0
LIN = np.where(_c <= 0.04045, _c / 12.92, ((_c + 0.055) / 1.055) ** 2.4).astype(np.float32)
M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.793617785, -0.0040720468],
               [1.9779984951, -2.428592205, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.808675766]])
WEIGHT = np.array([1.3, 1.0, 1.0])  # lightness counts 1.3x in the colour distance


def oklab(lin):
    """Linear-light RGB (..., 3) -> OKLab (L in 0..1)."""
    return np.cbrt(np.asarray(lin, np.float64) @ M1.T) @ M2.T


def oklab_to_srgb_u8(lab):
    lms = np.asarray(lab, np.float64) @ np.linalg.inv(M2).T
    lin = np.clip((lms ** 3) @ np.linalg.inv(M1).T, 0, 1)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


def load_palette(which="40"):
    """'40' = the original PhotoBricks 40 (what the reference uses); 'current' = config/palette.json."""
    if str(which) == "40":
        return load_client_palette(CONFIG_DIR / "palette_40.json")
    return load_client_palette()


def palette_oklab(palette):
    return oklab(LIN[palette.rgb.astype(int)])


def _d2(points, pal):
    """Weighted squared OKLab distance (N, P)."""
    return (((points[:, None, :] - pal[None, :, :]) ** 2) * WEIGHT).sum(-1)


def _near(points, pal, subset):
    """Index (into the full palette) of the nearest colour in `subset` for each point.
    Ties go to the earlier colour, as in the original loop."""
    sub = np.asarray(subset)
    return sub[np.argmin(_d2(points, pal[sub]), axis=1)]


# ---------------------------------------------------------------- 1. crop

def keep_faces(faces):
    """Faces at least 30 % of the largest face's area (x, y, w, h in photo pixels)."""
    faces = [tuple(map(float, f[:4])) for f in faces]
    if not faces:
        return []
    big = max(w * h for _, _, w, h in faces)
    return [f for f in faces if f[2] * f[3] >= big * 0.3]


def crop_for(iw, ih, gw, gh, faces, auto_frame=True, zoom=1.0, dx=0.0, dy=0.0, plate=24):
    """Crop box (x, y, w, h) for a gw x gh-plate board. Smaller boards get a tighter crop on the
    face, since the face needs more studs to stay readable."""
    a = gw / gh
    h0 = min(ih, iw / a)
    ch, cx, cy = h0, iw / 2, ih / 2
    if faces and auto_frame:
        x0 = min(f[0] for f in faces)
        y0 = min(f[1] for f in faces)
        x1 = max(f[0] + f[2] for f in faces)
        y1 = max(f[1] + f[3] for f in faces)
        m = min(gw, gh) * plate
        fr = min(0.78, max(0.34, 0.78 - (m - 48) * 0.0037))
        ch = max((y1 - y0) / fr, (x1 - x0) * (1.5 if len(faces) > 1 else 1.12) / a)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2 + ch * 0.08
    ch = min(h0, ch / zoom)
    cw = ch * a
    cx += dx
    cy += dy
    return (min(max(cx - cw / 2, 0), iw - cw), min(max(cy - ch / 2, 0), ih - ch), cw, ch)


def _mitchell(x, B=1 / 3, C=1 / 3):
    x = np.abs(x)
    return np.where(x < 1, ((12 - 9 * B - 6 * C) * x ** 3 + (-18 + 12 * B + 6 * C) * x ** 2 + (6 - 2 * B)) / 6,
                    np.where(x < 2, ((-B - 6 * C) * x ** 3 + (6 * B + 30 * C) * x ** 2 + (-12 * B - 48 * C) * x
                                     + (8 * B + 24 * C)) / 6, 0.0))


def _resample_axis(a, start, length, n, axis):
    """Mitchell-Netravali cubic sampling along one axis (fixed 2-pixel support, edges clamped),
    the filter browsers use for high-quality canvas drawImage; mean error vs Chrome ~0.6 levels."""
    src = (np.arange(n) + 0.5) * (length / n) + start - 0.5
    base = np.floor(src).astype(int) - 1
    idx = base[:, None] + np.arange(4)[None, :]
    wts = _mitchell(idx - src[:, None])
    wts /= wts.sum(1, keepdims=True)
    idx = np.clip(idx, 0, a.shape[axis] - 1)
    a = np.moveaxis(a, axis, 0)
    out = np.einsum("nk,nk...->n...", wts, a[idx])
    return np.moveaxis(out, 0, axis)


def _resample(rgb, box, W, H):
    """Crop box (x, y, w, h) of the photo resampled to W x H, like canvas drawImage.
    Large reductions are first halved with a 2x2 box filter (mipmaps), as browsers do."""
    x, y, w, h = box
    img = rgb.astype(np.float64)
    while w / W >= 2 and h / H >= 2 and min(img.shape[:2]) >= 2:
        hh, ww = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        img = img[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2, 3).mean(axis=(1, 3))
        x, y, w, h = x / 2, y / 2, w / 2, h / 2
    t = _resample_axis(img, y, h, H, 0)
    t = _resample_axis(t, x, w, W, 1)
    return np.clip(np.rint(t), 0, 255).astype(np.uint8)


def _to_u8(v):
    """Store into a Uint8ClampedArray: round half to even, clamp."""
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 2. background, face brightness

def background(d, crop_rgb, box, faces, strength):
    """Blur and mute everything outside a soft head-and-shoulders oval around each face."""
    if not strength or not faces:
        return d
    H, W = d.shape[:2]
    x0, y0, cw, _ = box
    k = W / cw
    sh = max(2, round(40 * H / W))
    small = Image.fromarray(crop_rgb).resize((40, sh), Image.LANCZOS)
    B = np.asarray(small.resize((W, H), Image.BICUBIC)).astype(np.float64)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    m = np.zeros((H, W))
    for fx, fy, fw, fh in faces:
        cx, cy = (fx + fw / 2 - x0) * k, (fy + fh * 0.85 - y0) * k
        rx, ry = fw * 1.25 * k, fh * 1.9 * k
        e = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
        m = np.maximum(m, 1 - np.minimum(1, np.maximum(0, (e - 1) / 0.6)))
    b = ((1 - m) * strength)[..., None]
    yb = (B[..., 0] * 0.299 + B[..., 1] * 0.587 + B[..., 2] * 0.114)[..., None]
    target = B + (yb - B) * 0.5
    out = d.astype(np.float64)
    upd = b[..., 0] > 0
    out[upd] = out[upd] + (target[upd] - out[upd]) * b[upd]
    return _to_u8(out)


def face_gamma(d, box, faces):
    """Gamma that lifts the face to a pleasant skin brightness (never darkens): 0.6 .. 1."""
    if not faces:
        return 1.0
    H, W = d.shape[:2]
    x0c, y0c, cw, _ = box
    k = W / cw
    total, n = 0.0, 0
    luma = d[..., 0] * 0.299 + d[..., 1] * 0.587 + d[..., 2] * 0.114
    for fx, fy, fw, fh in faces:
        xa = max(0, int((fx - x0c) * k))
        xb = min(W, int((fx + fw - x0c) * k))
        ya = max(0, int((fy - y0c) * k))
        yb = min(H, int((fy + fh - y0c) * k))
        if xb > xa and yb > ya:
            total += float(luma[ya:yb, xa:xb].sum())
            n += (xb - xa) * (yb - ya)
    if not n:
        return 1.0
    mean = min(0.95, max(0.05, total / n / 255))
    return min(1.0, max(0.6, np.log(0.68) / np.log(mean)))


# ---------------------------------------------------------------- 3. tone

TARGET_CHROMA = 28.0  # median CIELAB chroma that looks right in bricks (a sepia portrait measures ~29)


def auto_saturation(d):
    """Saturation for this photo from how colourful it already is: dull photos are boosted, vivid
    ones held back. Half the gap to TARGET_CHROMA, limited to -20 .. +30."""
    from .colour import rgb2lab
    small = d[::4, ::4].astype(np.float64) / 255.0
    lab = rgb2lab(small)
    c = float(np.median(np.hypot(lab[..., 1], lab[..., 2])))
    return round(float(np.clip(50 * (TARGET_CHROMA / max(c, 1.0) - 1), -20, 30)), 1), round(c, 1)

def tone(d, contrast, saturation, gamma, auto_levels):
    lo, hi = 0, 255
    if auto_levels:
        luma = (d[..., 0] * 0.299 + d[..., 1] * 0.587 + d[..., 2] * 0.114).astype(np.int64)
        hist = np.bincount(luma.ravel(), minlength=256)
        n = luma.size
        cum = np.cumsum(hist)
        lo = int(min(np.argmax(cum >= n * 0.01), 255))
        rcum = np.cumsum(hist[::-1])
        hi = int(max(255 - np.argmax(rcum >= n * 0.01), 0))
        if hi - lo < 30:
            lo, hi = 0, 255
        lo, hi = min(lo, 40), max(hi, 215)
    k = 255 / (hi - lo)
    c, s = 1 + contrast / 100, 1 + saturation / 100
    lut = _to_u8(255 * (np.arange(256) / 255) ** gamma)
    rgb = (d.astype(np.float64) - lo) * k
    y = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
    v = (y + (rgb - y) * s - 128) * c + 128
    return lut[np.minimum(255, np.maximum(0, v)).astype(np.int64)]


# ---------------------------------------------------------------- 4-5. stud grid in OKLab

def grid(d, GW, GH):
    lin = LIN[d].astype(np.float64).reshape(GH, CELL, GW, CELL, 3).mean(axis=(1, 3))
    return oklab(lin).astype(np.float32)


def _shift(a, dy, dx):
    """a[y+dy, x+dx] with a validity mask (out-of-range neighbours are skipped)."""
    H, W = a.shape[:2]
    out = np.zeros_like(a)
    valid = np.zeros((H, W), bool)
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[yd, xd] = a[ys, xs]
    valid[yd, xd] = True
    return out, valid


def smooth(o, s):
    """Edge-preserving (bilateral) smoothing in OKLab, radius 2."""
    if not s:
        return o
    w = o.astype(np.float64)
    acc = np.zeros_like(w)
    tot = np.zeros(w.shape[:2])
    for j in range(-2, 3):
        for i in range(-2, 3):
            n, v = _shift(w, j, i)
            dd = ((n - w) ** 2).sum(-1)
            q = np.exp(-dd / 0.0036 - (i * i + j * j) / 4) * v
            acc += n * q[..., None]
            tot += q
    return (w + (acc / tot[..., None] - w) * s).astype(np.float32)


def blur_l(o, r):
    """Box blur of L with windows clamped at the edges."""
    L = o[..., 0].astype(np.float32)
    H, W = L.shape
    acc = np.zeros((H, W))
    cnt = np.zeros((H, W))
    Lf = L.astype(np.float64)
    for i in range(-r, r + 1):
        n, v = _shift(Lf, 0, i)
        acc += n
        cnt += v
    T = (acc / cnt).astype(np.float32).astype(np.float64)
    acc[:] = 0
    cnt[:] = 0
    for j in range(-r, r + 1):
        n, v = _shift(T, j, 0)
        acc += n
        cnt += v
    return L, (acc / cnt).astype(np.float32)


def local_contrast(o, a):
    if not a:
        return o
    GH, GW = o.shape[:2]
    L, B = blur_l(o, max(3, round(min(GW, GH) / 10)))
    o = o.copy()
    o[..., 0] = (o[..., 0].astype(np.float64) + a * (L.astype(np.float64) - B)).astype(np.float32)
    return o


def dark_lines(o, a):
    """Darken only studs darker than their surroundings: brows, eyes, nostrils, mouth, beard edges."""
    if not a:
        return o
    L, B = blur_l(o, 2)
    o = o.copy()
    o[..., 0] = (o[..., 0].astype(np.float64)
                 + a * np.maximum(-0.25, np.minimum(0, L.astype(np.float64) - B))).astype(np.float32)
    return o


def sharpen(o, a):
    if not a:
        return o
    L = o[..., 0].astype(np.float64)
    up = np.vstack([L[:1], L[:-1]])
    down = np.vstack([L[1:], L[-1:]])
    left = np.hstack([L[:, :1], L[:, :-1]])
    right = np.hstack([L[:, 1:], L[:, -1:]])
    o = o.copy()
    o[..., 0] = (L + a * (4 * L - up - down - left - right)).astype(np.float32)
    return o


# ---------------------------------------------------------------- 6-7. palette and assignment

def reduce_palette(points, pal, max_colours):
    """Drop the colour whose loss hurts least (count x distance to its nearest neighbour) until
    at most max_colours remain and none is used by fewer than 0.8 % of studs (min 6 colours)."""
    N = len(points)
    subset = list(range(len(pal)))
    min_n = N * 0.008
    cnt = dict.fromkeys(subset, 0)
    for i, c in zip(*np.unique(_near(points, pal, subset), return_counts=True)):
        cnt[int(i)] = int(c)
    D = _d2(pal, pal)
    while len(subset) > 6:
        worst, wc = -1, np.inf
        for i in subset:
            if not cnt[i]:
                worst = i
                break
            md = min(D[i, j] for j in subset if j != i)
            c = cnt[i] * md
            if c < wc:
                wc, worst = c, i
        if len(subset) <= max_colours and cnt[worst] >= min_n:
            break
        rest = [i for i in subset if i != worst]
        nb = int(_near(pal[worst][None], pal, rest)[0])
        cnt[nb] += cnt[worst]
        subset = rest
    return subset


def assign(o, pal, subset, dither):
    GH, GW = o.shape[:2]
    if not dither:
        return _near(o.reshape(-1, 3).astype(np.float64), pal, subset).reshape(GH, GW)
    w = o.copy()
    sub = np.asarray(subset)
    P = pal[sub]
    out = np.zeros((GH, GW), np.int64)
    taps = ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
    for y in range(GH):
        for x in range(GW):
            p = w[y, x].astype(np.float64)
            i = sub[int(np.argmin((((P - p) ** 2) * WEIGHT).sum(-1)))]
            out[y, x] = i
            e = p - pal[i]
            for dx, dy, f in taps:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GW and ny < GH:
                    w[ny, nx] = (w[ny, nx].astype(np.float64) + e * f * dither).astype(np.float32)
    return out


# ---------------------------------------------------------------- pipeline

@lru_cache(maxsize=4)
def _photo(path, mtime, score):
    rgb = load_image(path)
    return rgb, keep_faces(detect_faces(rgb, score))


def photo_and_faces(path, score=0.8):
    p = Path(path)
    return _photo(str(p), p.stat().st_mtime, score)


def board_crop(rgb, faces, cols, rows, s, plate=24):
    """Explicit crop from settings ([x, y, w, h] or [x, y, side]) or the reference auto-frame."""
    ih, iw = rgb.shape[:2]
    crop = s.get("crop")
    if crop in (None, "auto"):
        return crop_for(iw, ih, cols / plate, rows / plate, faces, s.get("studio_auto_frame", True), plate=plate)
    if len(crop) == 3:
        x, y, side = crop
        crop = [x, y, side, side] if cols == rows else [x, y, side * cols / rows, side]
    x, y, w, h = map(float, crop)
    h = w * rows / cols  # keep the board's shape exactly
    w, h = min(w, iw, ih * cols / rows), min(h, ih, iw * rows / cols)
    return (min(max(x, 0), iw - w), min(max(y, 0), ih - h), w, h)


def convert(path, size, s, dither=0.0, work=None, faces=None):
    """Returns (grid codes, used palette, pre dict, target sRGB per stud).

    work: optional board image already cropped and resampled to 8 px per stud (the page does
    this with the browser's canvas, exactly as the reference does); s["crop"] is then its crop
    box [x, y, w, h] in photo pixels and faces are in photo pixels too."""
    cols, rows = (size, size) if isinstance(size, (int, np.integer)) else size
    W, H = cols * CELL, rows * CELL
    if work is None:
        rgb, found = photo_and_faces(path, s.get("face_score", 0.8))
        faces = found if faces is None else keep_faces(faces)
        box = board_crop(rgb, faces, cols, rows, s)
        d = _resample(rgb, box, W, H)
        x, y, w, h = box
        xi, yi, wi, hi = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        crop = rgb[yi:yi + hi, xi:xi + wi].copy()
    else:
        if work.shape[:2] != (H, W):
            raise ValueError(f"work image must be {W}x{H}, got {work.shape[1]}x{work.shape[0]}")
        d = np.ascontiguousarray(work[..., :3])
        faces = keep_faces(faces or [])
        box = tuple(map(float, s["crop"]))
        crop = d
    crop_view = d.copy()
    d = background(d, crop_view, box, faces, s.get("studio_bg_simplify", 0.7))
    gam = face_gamma(d, box, faces) if s.get("studio_brighten_face", True) else 1.0
    sat = s.get("studio_saturation", -2)
    if s.get("studio_auto_saturation", False):
        sat, chroma = auto_saturation(crop_view)
        s["studio_saturation_used"], s["photo_chroma"] = sat, chroma
    d = tone(d, s.get("studio_contrast", 10), sat, gam, s.get("studio_auto_levels", True))
    o = grid(d, cols, rows)
    o = smooth(o, s.get("studio_smoothing", 0.6))
    o = local_contrast(o, s.get("studio_local_contrast", 0.6))
    o = dark_lines(o, s.get("studio_dark_lines", 1.0))
    o = sharpen(o, s.get("studio_detail", 0.1))

    palette = load_palette(s.get("studio_palette", "40"))
    pal = palette_oklab(palette)
    pts = o.reshape(-1, 3).astype(np.float64)
    subset = reduce_palette(pts, pal, int(s.get("studio_colors", 40)))
    idx = assign(o, pal, subset, float(dither or 0))

    grid_codes = [[palette.codes[i] for i in row] for row in idx.tolist()]
    used = Palette(palette.name, palette.version, [palette.entries[i] for i in subset])
    x, y, w, h = box
    k = crop.shape[1] / w  # crop image pixels per photo pixel
    crop_faces = [((fx - x) * k, (fy - y) * k, fw * k, fh * k, 1.0) for fx, fy, fw, fh in faces
                  if x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h]
    pre = {
        "crop": crop,
        "crop_box": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
        "faces": [[round(v, 1) for v in f[:4]] for f in crop_faces],
        "face_mask": face_mask(crop.shape, crop_faces),
        "image": crop,
    }
    cells = oklab_to_srgb_u8(o.astype(np.float64))
    return grid_codes, used, pre, cells
