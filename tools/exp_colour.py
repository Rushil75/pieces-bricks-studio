"""Colour-usage experiments: saturation boost / chroma weight vs Brick.me (run from project root)."""
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from benchmark import extract_grid, score  # noqa: E402
from engine import export, pipeline, render  # noqa: E402
from engine.colour import rgb2lab  # noqa: E402
from engine.palette import hex_to_rgb  # noqa: E402

OUT = sys.argv[1]
VARIANTS = [("current", {}), ("sat +15", {"saturation_adj": 15}), ("sat +25", {"saturation_adj": 25}),
            ("kC 0.7", {"kC": 0.7}), ("sat +20, kC 0.8", {"saturation_adj": 20, "kC": 0.8})]
CASES = [("saint", {"file": "saint.jpg", "crop": [0, 0, 554]}, 96), ("p5", None, 48), ("p1", None, 48)]
font = ImageFont.truetype("engine/fonts/DejaVuSans-Bold.ttf", 15)
T = 384
rows = []
for job, base, n in CASES:
    row = []
    if job == "saint":
        shot = Image.open("input/ref_brickme_saint.png").convert("RGB").crop((717, 105, 1487, 876)).resize((T, T), Image.LANCZOS)
        d = ImageDraw.Draw(shot)
        d.rectangle([0, 0, T, 20], fill=(0, 0, 0))
        d.text((4, 2), "Brick.me", fill="white", font=font)
        row.append(shot)
    for label, o in VARIANTS:
        s = pipeline.default_settings({**base, **o}) if base else pipeline.job_settings(job, o)
        grid, pal, pre = pipeline.convert(pipeline.ROOT / "input" / s["file"], n, False, s)
        m = export.build_matrix(grid, pal, {})
        hexes = {e["code"]: hex_to_rgb(e["hex"]) for e in pal.entries}
        g = np.array([[hexes[c] for c in r] for r in grid], np.uint8)
        ref = cv2.resize(pre["crop"], (n, n), interpolation=cv2.INTER_AREA)
        face = cv2.resize(pre["face_mask"], (n, n), interpolation=cv2.INTER_AREA) > 0.5
        sc = score(g, ref, face)
        lab = rgb2lab(g / 255.0)
        chroma = np.percentile(np.hypot(lab[..., 1], lab[..., 2]), 50)
        cnt = Counter(np.array(grid).ravel())
        im = render.render_preview(m, T // n)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, T, 20], fill=(0, 0, 0))
        d.text((4, 2), f"{label}: C* {chroma:.0f}, {len(cnt)} col", fill="white", font=font)
        row.append(im)
        print(f"{job:5s} {label:18s} chroma p50 {chroma:5.1f}  colours {len(cnt):2d}  red wine {cnt.get('C37', 0) / g.shape[0] / g.shape[1] * 100:4.1f}%  "
              f"view dE {sc['view dE']:5.2f}  face SSIM {sc.get('face SSIM', 0):5.1f}  grain {sc['grain']:5.1f}")
    pipeline._preprocess_cached.cache_clear()
    rows.append(row)
cols = max(len(r) for r in rows)
out = Image.new("RGB", (cols * (T + 8), len(rows) * (T + 8)), "white")
for j, r in enumerate(rows):
    off = 0 if len(r) == cols else 1
    for i, im in enumerate(r):
        out.paste(im, ((i + off) * (T + 8), j * (T + 8)))
out.save(OUT)
