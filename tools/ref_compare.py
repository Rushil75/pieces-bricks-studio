"""Compare the Studio engine with grids captured from the reference page (tools/ref_capture.py).

  python tools/ref_compare.py CAPTURE.json PHOTO [json-settings]

Uses the reference's own crop and faces, so only the image processing is compared.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from engine import pipeline, studio  # noqa: E402

cap = json.load(open(sys.argv[1]))
photo = sys.argv[2]
over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
faces = [(f["x"], f["y"], f["w"], f["h"]) for f in cap["faces"]]
rgb = studio.load_image(photo)
studio.photo_and_faces = lambda path, score=0.8: (rgb, faces)  # use the reference's faces
codes = studio.load_palette("40").codes
for b in cap["boards"]:
    w, h = b["plates"]
    cols, rows = w * 24, h * 24
    c = b["crop"]
    s = pipeline.default_settings({"engine": "studio", "crop": [c["x"], c["y"], c["w"], c["h"]], **over})
    grid, used, pre, cells = studio.convert(photo, (cols, rows), s, s.get("studio_dither", 0))
    ours = np.vectorize(codes.index)(np.array(grid))
    ref = np.array(b["idx"]).reshape(rows, cols)
    same = (ours == ref).mean() * 100
    print(f"{w}x{h} plates ({cols}x{rows}): same colour {same:5.1f}%   colours ours {len(np.unique(ours))} / ref {len(np.unique(ref))}")
