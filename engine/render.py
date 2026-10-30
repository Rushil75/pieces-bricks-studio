"""preview.png, flat.png, grid.png, compare.png -- all drawn from a matrix.json dict."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .palette import hex_to_rgb

FONT = str(Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf")  # bundled: hosts lack system fonts
FONT_BOLD = str(Path(__file__).resolve().parent / "fonts" / "DejaVuSans-Bold.ttf")
GAP = (24, 22, 26)


def cell_sizes(n):
    """(preview, flat, grid) pixels per brick for an n x n board, so files stay a sensible size."""
    if n <= 64:
        return 22, 10, 30
    if n <= 128:
        return 12, 6, 22
    if n <= 192:
        return 8, 4, 18
    return 8, 4, 15  # keeps the 256x256 download pack under serverless 4.5 MB responses


def _font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _colors(matrix):
    return {e["code"]: tuple(hex_to_rgb(e["hex"])) for e in matrix["palette"]["colors"]}


def _mix(c, t, a):
    return tuple(int(round(c[i] * (1 - a) + t[i] * a)) for i in range(3))


def _brick_tile(rgb, cell, ss=4):
    """One brick with a round stud, drawn supersampled then reduced for smooth edges."""
    s = cell * ss
    # Bricks touch like real ones: the seam is a darker shade of the brick's own colour (no black
    # grid), and the stud's shadow and highlight give the 3D look. Same rules as web/src/bricks.ts.
    k = min(1.0, max(0.35, (cell - 4) / 16))
    g = max(1, round(s * 0.025))
    bev = max(1, round(s * 0.04))
    im = Image.new("RGB", (s, s), _mix(rgb, (0, 0, 0), 0.35))
    d = ImageDraw.Draw(im)
    light, dark = _mix(rgb, (255, 255, 255), 0.12 * k), _mix(rgb, (0, 0, 0), 0.12 * k)
    d.rectangle([g, g, s - g - 1, s - g - 1], fill=dark)                   # bottom/right shade
    d.rectangle([g, g, s - g - 1 - bev, s - g - 1 - bev], fill=light)      # top/left bevel
    d.rectangle([g + bev, g + bev, s - g - 1 - bev, s - g - 1 - bev], fill=rgb)
    c, r = s / 2, s * 0.3
    off = s * 0.045 * k
    d.ellipse([c - r + off, c - r + off, c + r + off, c + r + off], fill=_mix(rgb, (0, 0, 0), 0.3 * k))  # stud shadow
    d.ellipse([c - r, c - r, c + r, c + r], fill=_mix(rgb, (0, 0, 0), 0.1 * k))                          # stud side
    ri, o = r * 0.86, off * 0.4
    d.ellipse([c - ri - o, c - ri - o, c + ri - o, c + ri - o], fill=_mix(rgb, (255, 255, 255), 0.06 * k))  # stud top
    d.arc([c - ri, c - ri, c + ri, c + ri], 190, 265,
          fill=_mix(rgb, (255, 255, 255), 0.5 * k), width=max(1, round(s * 0.05)))                         # highlight
    return im.resize((cell, cell), Image.LANCZOS)


def render_preview(matrix, cell=22):
    n, cols = matrix["size"], _colors(matrix)
    tiles = {code: _brick_tile(rgb, cell) for code, rgb in cols.items()}
    out = Image.new("RGB", (n * cell, n * cell), GAP)
    for y, row in enumerate(matrix["grid"]):
        for x, code in enumerate(row):
            out.paste(tiles[code], (x * cell, y * cell))
    return out


def render_flat(matrix, cell=10):
    cols = _colors(matrix)
    arr = np.array([[cols[c] for c in row] for row in matrix["grid"]], np.uint8)
    return Image.fromarray(arr).resize((matrix["size"] * cell,) * 2, Image.NEAREST)


def render_grid(matrix, cell=30, block=8):
    n, cols = matrix["size"], _colors(matrix)
    margin = cell + 6
    W = margin * 2 + n * cell
    out = Image.new("RGB", (W, W), (255, 255, 255))
    d = ImageDraw.Draw(out)
    f, fe = _font(int(cell * 0.42), bold=True), _font(int(cell * 0.38))
    for y, row in enumerate(matrix["grid"]):
        for x, code in enumerate(row):
            rgb = cols[code]
            x0, y0 = margin + x * cell, margin + y * cell
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=rgb)
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            d.text((x0 + cell / 2, y0 + cell / 2), code[1:], fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=f, anchor="mm")
    for i in range(n + 1):
        wdt, col = (3, (0, 0, 0)) if i % block == 0 else (1, (90, 90, 90))
        p = margin + i * cell
        d.line([p, margin, p, margin + n * cell], fill=col, width=wdt)
        d.line([margin, p, margin + n * cell, p], fill=col, width=wdt)
    for i in range(n):
        c = margin + i * cell + cell / 2
        for pos in ((c, margin / 2), (c, W - margin / 2)):
            d.text(pos, str(i + 1), fill=(0, 0, 0), font=fe, anchor="mm")
        for pos in ((margin / 2, c), (W - margin / 2, c)):
            d.text(pos, str(i + 1), fill=(0, 0, 0), font=fe, anchor="mm")
    return out


def render_compare(original, panels, caption):
    """original crop + labelled previews side by side, caption underneath.
    Previews stay at native size: resampling brick gaps turns them into a grey haze."""
    pad, head, foot = 24, 46, 60
    height = panels[0][1].height
    imgs = [("Original crop", original.resize((height, height), Image.LANCZOS))] + panels
    W = pad + len(imgs) * (height + pad)
    out = Image.new("RGB", (W, height + head + foot + pad), (250, 248, 244))
    d = ImageDraw.Draw(out)
    ft, fc = _font(26, bold=True), _font(24)
    for i, (title, im) in enumerate(imgs):
        x = pad + i * (height + pad)
        d.text((x, pad / 2 + 4), title, fill=(30, 30, 30), font=ft)
        out.paste(im, (x, head))
    d.text((pad, head + height + 18), caption, fill=(60, 60, 60), font=fc)
    return out
