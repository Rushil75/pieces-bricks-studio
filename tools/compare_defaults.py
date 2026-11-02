"""Render old vs new defaults on the test photos and print benchmark scores (run from project root)."""
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from benchmark import score  # noqa: E402
from engine import export, pipeline, render  # noqa: E402
from engine.palette import hex_to_rgb  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "compare_defaults.png"
OLD = {"kL": 1.0, "contrast": 15, "brightness": 5, "detail": 2.0, "blend": "off"}
import json
VARIANTS = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [["old (this morning)", OLD], ["new", {}], ["new, blend off", {"blend": "off"}]]
CASES = [("saint", {"file": "saint.jpg", "crop": [0, 0, 554]}, 96), ("p5", None, 48), ("p1", None, 48), ("p4", None, 48)]
font = ImageFont.truetype("engine/fonts/DejaVuSans-Bold.ttf", 15)
rows = []
for job, base, n in CASES:
    row = []
    for label, o in VARIANTS:
        s = pipeline.default_settings({**base, **o}) if base else pipeline.job_settings(job, o)
        t0 = time.time()
        grid, pal, pre = pipeline.convert(pipeline.ROOT / "input" / s["file"], n, False, s)
        dt = time.time() - t0
        m = export.build_matrix(grid, pal, {})
        hexes = {e["code"]: hex_to_rgb(e["hex"]) for e in pal.entries}
        g = np.array([[hexes[c] for c in r] for r in grid], np.uint8)
        ref = cv2.resize(pre["crop"], (n, n), interpolation=cv2.INTER_AREA)
        face = cv2.resize(pre["face_mask"], (n, n), interpolation=cv2.INTER_AREA) > 0.5
        sc = score(g, ref, face)
        im = render.render_preview(m, 384 // n)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 384, 20], fill=(0, 0, 0))
        d.text((4, 2), f"{job} {label}: view dE {sc['view dE']:.2f} grain {sc['grain']:.1f}", fill="white", font=font)
        row.append(im)
        print(f"{job:5s} {label:20s} view dE {sc['view dE']:5.2f}  view SSIM {sc['view SSIM']:5.1f}  grain {sc['grain']:5.1f}  "
              f"face SSIM {sc.get('face SSIM', 0):5.1f}  face view dE {sc.get('face view dE', 0):5.2f}  {dt:.2f}s")
    pipeline._preprocess_cached.cache_clear()
    rows.append(row)
out = Image.new("RGB", (3 * 394, len(rows) * 394), "white")
for j, r in enumerate(rows):
    for i, im in enumerate(r):
        out.paste(im, (i * 394, j * 394))
out.save(OUT)
