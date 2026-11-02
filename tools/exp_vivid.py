"""Reproduce Brick.me's colour treatment on the saint and compare with ours (run from project root).

  python tools/exp_vivid.py OUT_PREFIX
"""
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from benchmark import extract_grid, score  # noqa: E402
from engine import pipeline, render  # noqa: E402
from engine.colour import rgb2lab  # noqa: E402
from engine.palette import ciede2000_matrix, load_client_palette, select_subset  # noqa: E402
from engine.quantize import match  # noqa: E402

OUT = sys.argv[1]
N = 96
client = load_client_palette()
names = {e["code"]: e["name"] for e in client.entries}
bm = extract_grid("input/ref_brickme_saint.png", (717, 105, 1487, 876), N, N)
bm_idx = ciede2000_matrix(rgb2lab(bm / 255.0).reshape(-1, 3), client.lab, 1.0).argmin(1).reshape(N, N)
valid = np.ones((N, N), bool)
for r in range(N):
    for c in range(N):
        x, y = 717 + (c + 0.5) * 770 / N, 105 + (r + 0.5) * 771 / N
        valid[r, c] = not (y > 840 and 980 < x < 1225)


def brickme_transform(rgb, sat=1.2, con=1.2, gamma=0.9, sharpen=1.0):
    """The transform that best reproduces Brick.me's brick choices (found by search)."""
    x = np.clip(0.5 + (rgb.astype(np.float32) / 255.0 - 0.5) * con, 0, 1) ** gamma
    hsv = cv2.cvtColor((x * 255).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
    y = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if sharpen:
        y = y + (y - cv2.GaussianBlur(y, (0, 0), 1.0)) * sharpen
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def run(over, transform=None):
    s = pipeline.default_settings({"file": "saint.jpg", "crop": [0, 0, 554], **over})
    grid, pal, pre, cells = pipeline.convert("input/saint.jpg", N, False, s, return_cells=True)
    fm = cv2.resize(pre["face_mask"], (N, N), interpolation=cv2.INTER_AREA)
    if transform is None:
        idx = np.vectorize(client.codes.index)(np.array(grid))
    else:
        cells = transform(cells)
        w = 1.0 + (s["face_weight"] - 1.0) * np.clip(fm, 0, 1)
        sub = select_subset(cells, w, client, s["client_colors"], s["kL"])
        sidx = match(cells, sub, s["kL"])
        idx = np.vectorize(client.codes.index)(np.array(sub.codes)[sidx])
    ref = cv2.resize(pre["crop"], (N, N), interpolation=cv2.INTER_AREA)
    return idx, ref, fm > 0.5


VARIANTS = [
    ("ours default (filter None)", {}, None),
    ("ours, filter Vivid", {"filter": "vivid"}, None),
    ("ours, Vivid + Textured blend", {"filter": "vivid", "blend": "background"}, None),
]

font = ImageFont.truetype("engine/fonts/DejaVuSans-Bold.ttf", 16)
cell = 8
shot = Image.open("input/ref_brickme_saint.png").convert("RGB").crop((717, 105, 1487, 876)).resize((N * cell, N * cell), Image.LANCZOS)
boards = [("Brick.me", shot)]
bm_cnt = Counter(np.array(client.codes)[bm_idx[valid]])
rows = []
for name, over, tf in VARIANTS:
    idx, ref, face = run(over, tf)
    g = client.rgb[idx]
    sc = score(g, ref, face, valid)
    lab = rgb2lab(g / 255.0)
    chroma = float(np.percentile(np.hypot(lab[..., 1], lab[..., 2])[valid], 50))
    agree = float((idx == bm_idx)[valid].mean() * 100)
    cnt = Counter(np.array(client.codes)[idx[valid]])
    rows.append((name, agree, chroma, sc, cnt))
    m = {"palette": {"colors": client.entries}, "grid": [[client.codes[i] for i in r] for r in idx]}
    boards.append((name, render.render_preview(m, cell)))

bl = rgb2lab(bm / 255.0)
print(f"{'':38s} {'same as BM':>10s} {'chroma':>7s} {'view dE':>8s} {'face SSIM':>9s} {'grain':>6s}")
print(f"{'Brick.me':38s} {'':>10s} {np.percentile(np.hypot(bl[..., 1], bl[..., 2])[valid], 50):7.1f}")
for name, agree, chroma, sc, cnt in rows:
    print(f"{name:38s} {agree:9.1f}% {chroma:7.1f} {sc['view dE']:8.2f} {sc['face SSIM']:9.1f} {sc['grain']:6.1f}")
tot = valid.sum()
codes = sorted(set(bm_cnt) | set().union(*[r[4] for r in rows]), key=lambda k: -bm_cnt.get(k, 0))
print("\ncolour shares:  " + "  ".join(f"{n[:14]:>14s}" for n in ["Brick.me"] + [r[0] for r in rows]))
for code in codes:
    shares = [bm_cnt.get(code, 0)] + [r[4].get(code, 0) for r in rows]
    if max(shares) / tot < 0.008:
        continue
    print(f"{code} {names[code][:11]:11s}" + "".join(f"{v / tot * 100:15.1f}%" for v in shares))

# full boards and face close-ups
W = N * cell
y0, y1, x0, x1 = 24 * cell, 72 * cell, 22 * cell, 74 * cell
for suffix, box in (("full", (0, 0, W, W)), ("face", (x0, y0, x1, y1))):
    tiles = [(t, im.crop(box)) for t, im in boards]
    tw, th = tiles[0][1].size
    out = Image.new("RGB", (len(tiles) * (tw + 8), th + 24), "white")
    d = ImageDraw.Draw(out)
    for i, (t, im) in enumerate(tiles):
        d.text((i * (tw + 8), 3), t, fill="black", font=font)
        out.paste(im, (i * (tw + 8), 24))
    if suffix == "full":
        out = out.resize((out.width // 2, out.height // 2), Image.LANCZOS)
    out.save(f"{OUT}_{suffix}.png")
