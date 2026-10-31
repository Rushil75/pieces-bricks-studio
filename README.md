# PIECES & BRICKS – mosaic engine (demo)

Photo in, N×N brick colour matrix out, plus a brick preview, numbered grid and exact counts.

```bash
pip install -r requirements.txt
python run_demo.py --all                          # all jobs in config/photos.yaml, both variants + compare sheets
python run_demo.py --photo p5 --size 48 --dither off
python run_demo.py --photo p5 --palette client    # the client's fixed 58 colours (C01–C58)
python run_demo.py --photo p5 --colors 64         # adaptive palette with 64 colours
python -m pytest -q tests
```

## Web demo (Phase 2)

```bash
cd web && npm install && npm run build && cd ..   # once, or after front-end changes
uvicorn server:app --port 8000                     # then open http://127.0.0.1:8000
```

The page has: upload (or one of the 5 client photos), crop and zoom with a brick-grid overlay and a
face-width readout in bricks, board size (32×32 up to 256×256), the fixed Color 58 palette, face detail, background and
dithering. Generate runs the same engine as `run_demo.py`, so results match the offline set exactly.
The mosaic has Bricks, Flat, Numbered grid and Compare (a split slider against the photo) views. You
can paint or pick a brick's colour, undo and redo, and see live brick counts with a total check. The
download pack (zip) is rendered on the server from the edited matrix.

For front-end development, run `npm run dev` in `web/` (it proxies `/api` to port 8000).

## Hosting on Vercel (one project)

Vercel runs `server.py` (FastAPI) as one Python function, which serves both the page
(`web/dist`, committed) and the `/api` routes, the same as `uvicorn server:app` locally.

1. Vercel > Add New > Project > import this repo. Keep Root Directory `./` and choose the
   **FastAPI** preset (or "Import single project" on the `app` FastAPI entry).
2. Optional environment variable: `DEMO_PASSWORD` (the page then asks for it).
3. Deploy.

Fitting the serverless limits:
- Dependencies are kept to about 250 MB: plain `opencv-python-headless`, and colour maths in
  `engine/colour.py` (NumPy) instead of scikit-image and SciPy. `tests/test_colour.py` checks it
  against scikit-image (`pip install -r requirements-dev.txt`).
- The API is stateless: the page sends the photo with every conversion, shrunk in the browser
  to at most 3000 px and 4 MB, because requests are capped at 4.5 MB.
- After changing the front end, run `npm run build` in `web/` and commit `web/dist`.

## Editor options (like the reference product)

Adjust: Detail, Face detail, Dither (0-100), Brightness, Contrast, Saturation. Filters: None, B&W,
Sepia, Warm, Cool, Vivid. Changes regenerate the mosaic automatically (not after painting, so
edits are kept). Colours: the fixed Color 58 palette, and a × per colour in the counts table, which
moves its bricks to the nearest colour still in use (CIEDE2000, same as the engine; undoable).
Defaults are tuned for faithful portraits: Detail 1, everything else neutral.
- The client photos are not in the repo, so the "try a client photo" buttons only appear where
  `input/` holds them. Uploads always work.

## Similarity score

Every result records `similarity` in matrix.json, and the CLI and the page show it. The reference is
the untouched crop averaged under each brick: the most detail an N×N board can hold. The score is
50 % the share of bricks within CIEDE2000 ΔE ≤ 5 of that colour, plus 50 % structural similarity
(SSIM on lightness). Mean ΔE, the share under ΔE 2.3 (no visible difference) and the face-only ΔE
are stored too. Tuned for the client's rules (48×48, at most 60 colours): single-person photos score 95.6–97.1 % without dithering; group photos 90–95 %.

## Palette modes

- `client` (default, and the only mode the web page uses): the 58 colours from `config/palette.json`, codes `C01–C58`.
  Every brick gets the nearest of these by CIEDE2000.
- `adaptive` (CLI only, `--palette adaptive`): picks the best `colors` (default 60, the client's limit; the engine allows up to 256) for each photo with weighted k-means in
  LAB. Face cells count more, and 20 % of the colour budget goes to the worst-matched cells, so small
  details such as eyes and lips keep their real colour. Codes are `A01…`, ordered dark to light. Each
  colour also lists its nearest client colour.

## Pipeline

1. EXIF-aware load. 2. YuNet face detection. 3. Square crop (auto from faces or manual `[x, y, size]`).
4. FSRCNN ×4 upscale when the crop is under 1000 px. 5. Light denoise and face-weighted unsharp (CLAHE
optional; off by default because it adds dark halos). 6. Background flattening outside a GrabCut person
mask. 7. `INTER_AREA` downscale, then brick-scale lightness sharpening inside faces.
8. CIEDE2000 matching, optionally with serpentine Floyd–Steinberg in LAB (weaker on faces).

Everything is deterministic. Each output folder `output/<job>_<size>_<variant>/` has `matrix.json`
(the source of truth, including every setting used), plus `preview.png`, `flat.png`, `grid.png`,
`counts.csv` and `crop.png`, all rendered from `matrix.json`.
