# PIECES & BRICKS – mosaic engine (demo)

Photo in, N×N brick colour matrix out, plus a brick preview, numbered grid and exact counts.

```bash
pip install -r requirements.txt
python run_demo.py --all                          # all jobs in config/photos.yaml, both variants + compare sheets
python run_demo.py --photo p5 --size 48 --dither off
python run_demo.py --photo p5 --palette client    # the client's fixed 40 colours (C01–C40)
python run_demo.py --photo p5 --colors 64         # adaptive palette with 64 colours
python -m pytest -q tests
```

## Web demo (Phase 2)

```bash
cd web && npm install && npm run build && cd ..   # once, or after front-end changes
uvicorn server:app --port 8000                     # then open http://127.0.0.1:8000
```

The page has: upload (or one of the 5 client photos), crop and zoom with a brick-grid overlay and a
face-width readout in bricks, board size (32×32 up to 256×256), palette mode, number of colours, face detail, background and
dithering. Generate runs the same engine as `run_demo.py`, so results match the offline set exactly.
The mosaic has Bricks, Flat, Numbered grid and Compare (a split slider against the photo) views. You
can paint or pick a brick's colour, undo and redo, and see live brick counts with a total check. The
download pack (zip) is rendered on the server from the edited matrix.

For front-end development, run `npm run dev` in `web/` (it proxies `/api` to port 8000).

## Hosting: page on Netlify, API on Render

Netlify serves static pages only. The Python engine (OpenCV, scikit-image) runs as a Docker
service on Render, and the page calls it directly.

1. **API on Render**: render.com > New > Blueprint > pick this repo (it reads `render.yaml`).
   Set `ALLOWED_ORIGINS` to the Netlify URL (you can fill it in after step 2), and optionally
   `DEMO_PASSWORD`. Check that `https://<service>.onrender.com/api/health` returns `{"ok": true}`.
2. **Page on Netlify**: Add new site > Import from Git > pick this repo and the branch. `netlify.toml`
   sets the build. In Site configuration > Environment variables, add
   `VITE_API_BASE = https://<service>.onrender.com`, then redeploy.
3. Put the Netlify URL in Render's `ALLOWED_ORIGINS` and let it redeploy.

The client photos are not in the repo. To offer the "try a client photo" buttons on the hosted
demo, copy them into `input/` in a private fork or on the server. Uploads work without them.
Render's free plan sleeps after 15 minutes idle, so the page shows "Starting the mosaic server…"
for up to a minute on the first visit. The free plan has 512 MB of memory, which is enough for
boards up to about 192×192; use a paid plan for 256×256 with dithering.

## Similarity score

Every result records `similarity` in matrix.json, and the CLI and the page show it. The reference is
the untouched crop averaged under each brick: the most detail an N×N board can hold. The score is
50 % the share of bricks within CIEDE2000 ΔE ≤ 5 of that colour, plus 50 % structural similarity
(SSIM on lightness). Mean ΔE, the share under ΔE 2.3 (no visible difference) and the face-only ΔE
are stored too. All current results score 95–98.5 % without dithering.

## Palette modes

- `adaptive` (default): picks the best `colors` (default 160, up to 256) for each photo with weighted k-means in
  LAB. Face cells count more, and 20 % of the colour budget goes to the worst-matched cells, so small
  details such as eyes and lips keep their real colour. Codes are `A01…`, ordered dark to light. Each
  colour also lists its nearest client colour.
- `client`: the 40 colours from `config/palette.json`, codes `C01–C40`.

## Pipeline

1. EXIF-aware load. 2. YuNet face detection. 3. Square crop (auto from faces or manual `[x, y, size]`).
4. FSRCNN ×4 upscale when the crop is under 1000 px. 5. Light denoise and face-weighted unsharp (CLAHE
optional; off by default because it adds dark halos). 6. Background flattening outside a GrabCut person
mask. 7. `INTER_AREA` downscale, then brick-scale lightness sharpening inside faces.
8. CIEDE2000 matching, optionally with serpentine Floyd–Steinberg in LAB (weaker on faces).

Everything is deterministic. Each output folder `output/<job>_<size>_<variant>/` has `matrix.json`
(the source of truth, including every setting used), plus `preview.png`, `flat.png`, `grid.png`,
`counts.csv` and `crop.png`, all rendered from `matrix.json`.
