import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import type { ConvertOptions, Crop, Filter, Matrix, PaletteColor, Photo } from "./api";
import { hexToLab, nearest } from "./colour";
import { setBrickStyle } from "./bricks";
import CropEditor, { reshapeCrop } from "./components/CropEditor";
import MosaicCanvas, { type Tool, type View } from "./components/MosaicCanvas";
import { PRINT_DEFAULTS, renderPrintSheet, sheetInfo, type PrintLayout } from "./print";
import { PLATE, studioCrop } from "./studioCrop";
import { detectFaces } from "./faces";
import SizeGrid, { BigView, posterPng, type SizeResult } from "./components/SizeGrid";

const STUD_MM = 8;
// Baseplates of 24 x 24 studs = 19.2 cm (as the reference); frames need +2 mm per side length.
const FRAME_MM = 2;
type Shape = "square" | "landscape" | "portrait" | "custom";
const SQUARE: [number, number][] = [2, 3, 4, 5, 6, 7].map((p) => [p, p]);
const PORTRAIT: [number, number][] = [[2, 2], [2, 3], [2, 4], [3, 3], [3, 4], [3, 5], [4, 4], [4, 5], [4, 6]];
const BOARDS: Record<Exclude<Shape, "custom">, [number, number][]> = {
  square: SQUARE,
  landscape: PORTRAIT.map(([w, h]) => [h, w]),
  portrait: PORTRAIT,
};
const MAX_PLATES = 10; // 240 studs, within the engine's 256 limit
const cm = (studs: number) => ((studs * STUD_MM) / 10).toFixed(1);
const frameCm = (studs: number) => ((studs * STUD_MM + FRAME_MM) / 10).toFixed(1);
const dimsOf = (m: Matrix): [number, number] => [m.grid[0].length, m.grid.length];
const FILTERS: [Filter, string][] = [
  ["none", "None"],
  ["bw", "B&W"],
  ["sepia", "Sepia"],
  ["warm", "Warm"],
  ["cool", "Cool"],
  ["vivid", "Vivid"],
];
const HISTORY = 100;

const DEFAULTS: ConvertOptions = {
  width: 48,
  height: 48,
  dither: 0,
  colors: 20, // best N of the client colours for each photo (like the reference product's "auto")
  blend: "off",
  detail: 1,
  brightness: 0,
  contrast: 0,
  saturation: 0,
  highlights: 0,
  filter: "none",
  face_detail: 0.15,
  background: "bilateral",
  engine: "studio",
  studio_palette: "40",
  studio_colors: 40,
  studio_contrast: 10,
  studio_saturation: -2,
  studio_smoothing: 0.6,
  studio_local_contrast: 0.6,
  studio_dark_lines: 1,
  studio_detail: 0.1,
  studio_dither: 0,
  studio_bg_simplify: 0.7,
  studio_auto_frame: true,
  studio_brighten_face: true,
  studio_auto_levels: true,
  studio_auto_saturation: false,
};

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function App() {
  const [samples, setSamples] = useState<{ key: string; title: string; job?: string }[]>([]);
  const [client, setClient] = useState<PaletteColor[]>([]);
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [photoBlob, setPhotoBlob] = useState<Blob | null>(null);
  const [job, setJob] = useState<string | undefined>(undefined);
  const [crop, setCrop] = useState<Crop>([0, 0, 1, 1]);
  const [opts, setOpts] = useState<ConvertOptions>(DEFAULTS);
  setBrickStyle(opts.engine === "studio" ? "studio" : "classic");
  const [shape, setShape] = useState<Shape>("square");
  const userCrop = useRef<Crop | null>(null); // last crop set by the user (or the auto crop)
  const autoCrop = useRef(true); // Studio: keep framing the faces per board until the user moves the crop
  // Studio: faces from the reference's own detector (in the browser); null = use the server's
  const [studioFaces, setStudioFaces] = useState<number[][] | null>(null);
  const [facesNote, setFacesNote] = useState("");
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [grid, setGrid] = useState<string[][] | null>(null);
  const [past, setPast] = useState<string[][][]>([]);
  const [future, setFuture] = useState<string[][][]>([]);
  const [generatedWith, setGeneratedWith] = useState<string>("");
  const [view, setView] = useState<View | "sizes">("sizes");
  const [sizeResults, setSizeResults] = useState<SizeResult[]>([]);
  const sizesReq = useRef(0);
  const [bigSize, setBigSize] = useState<SizeResult | null>(null);
  const [tool, setTool] = useState<Tool>("paint");
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState<string>("");
  const [error, setError] = useState<string>("");
  const fileInput = useRef<HTMLInputElement>(null);

  const [server, setServer] = useState<"waking" | "ready" | "down">("waking");
  const [needPw, setNeedPw] = useState(false);
  const [pwInput, setPwInput] = useState("");

  const loadBasics = useCallback(() => {
    api.listSamples().then(setSamples).catch((e) => e instanceof api.PasswordRequired && setNeedPw(true));
    api.clientPalette().then((p) => setClient(p.colors)).catch(() => {});
  }, []);

  // The hosted API sleeps when idle (Render free plan): poll until it answers.
  useEffect(() => {
    let stop = false;
    const started = Date.now();
    const ping = async () => {
      try {
        await api.health();
        if (stop) return;
        setServer("ready");
        loadBasics();
      } catch {
        if (stop) return;
        if (Date.now() - started > 120_000) setServer("down");
        else setTimeout(ping, 3000);
      }
    };
    ping();
    return () => {
      stop = true;
    };
  }, [loadBasics]);

  const fail = (e: unknown) => {
    if (e instanceof api.PasswordRequired) setNeedPw(true);
    else setError((e as Error).message);
  };

  const openPhoto = async (load: () => Promise<{ blob: Blob; job?: string }>) => {
    setError("");
    setBusy("Loading photo…");
    try {
      const src = await load();
      const blob = await api.preparePhoto(src.blob);
      const ph = await api.uploadPhoto(new File([blob], "photo.jpg", { type: blob.type }));
      const img = new Image();
      img.src = URL.createObjectURL(blob);
      await img.decode();
      if (image?.src.startsWith("blob:")) URL.revokeObjectURL(image.src);
      setPhoto(ph);
      setImage(img);
      setPhotoBlob(blob);
      setJob(src.job);
      userCrop.current = ph.auto_crop;
      autoCrop.current = true;
      // Board shape from the photo, as the reference: clearly tall = portrait, clearly wide = landscape
      setShape(ph.height > ph.width * 1.15 ? "portrait" : ph.width > ph.height * 1.15 ? "landscape" : "square");
      let faces = ph.faces;
      if (opts.engine === "studio") {
        setBusy("Finding faces…");
        faces = await findStudioFaces(img, ph);
      }
      setCrop(
        opts.engine === "studio"
          ? studioCrop({ ...ph, faces }, opts.width, opts.height, opts.studio_auto_frame)
          : reshapeCrop(ph.auto_crop, ph.width, ph.height, opts.width / opts.height),
      );
      setMatrix(null);
      setGrid(null);
      setPast([]);
      setFuture([]);
    } catch (e) {
      fail(e);
    } finally {
      setBusy("");
    }
  };

  /** Faces for the Studio engine: the reference's detector in the browser, else the server's. */
  const findStudioFaces = async (img: HTMLImageElement, ph: Photo): Promise<number[][]> => {
    try {
      const f = await detectFaces(img);
      setStudioFaces(f);
      setFacesNote(f.length ? `${f.length} face${f.length > 1 ? "s" : ""} found` : "No face found: frame the photo by hand.");
      return f;
    } catch {
      setStudioFaces(null);
      setFacesNote("Browser face detector unavailable: using the server's.");
      return ph.faces;
    }
  };
  const framePhoto = photo && opts.engine === "studio" && studioFaces ? { ...photo, faces: studioFaces } : photo;

  /** Crop for any board: the Studio framing per size, or the user's crop reshaped. */
  const cropForBoard = (w: number, h: number): Crop => {
    if (!photo) return crop;
    if (opts.engine === "studio" && autoCrop.current && framePhoto) return studioCrop(framePhoto, w, h, opts.studio_auto_frame);
    return reshapeCrop(userCrop.current ?? crop, photo.width, photo.height, w / h);
  };

  /** Mosaics for every size of the current shape, like the reference demo (latest request wins). */
  const generateAllSizes = async () => {
    if (!photo || !photoBlob || shape === "custom") return setSizeResults([]);
    const id = ++sizesReq.current;
    const list = BOARDS[shape];
    const results: SizeResult[] = list.map((p) => ({ plates: p, matrix: null }));
    setSizeResults([...results]);
    for (let i = 0; i < list.length; i++) {
      const [pw, ph] = list[i];
      const o = { ...opts, width: pw * PLATE, height: ph * PLATE };
      const c = cropForBoard(o.width, o.height);
      try {
        const m =
          o.engine === "studio" && image
            ? await api.convertStudio(image, photo.width, c, framePhoto?.faces ?? photo.faces, o)
            : await api.convert(photoBlob, c, o, job);
        if (id !== sizesReq.current) return;
        results[i] = { plates: list[i], matrix: m };
        setSizeResults([...results]);
      } catch (e) {
        if (id === sizesReq.current) fail(e);
        return;
      }
    }
  };

  // Switching Square / Landscape / Portrait refreshes the size cards.
  useEffect(() => {
    if (matrix) generateAllSizes();
  }, [shape]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Open one size from the grid for editing. */
  const openSize = (r: SizeResult) => {
    if (!r.matrix) return;
    const [pw, ph] = r.plates;
    const o = { ...opts, width: pw * PLATE, height: ph * PLATE };
    const c = cropForBoard(o.width, o.height);
    setOpts(o);
    setCrop(c);
    setMatrix(r.matrix);
    setGrid(r.matrix.grid);
    setPast([]);
    setFuture([]);
    setGeneratedWith(JSON.stringify({ id: photo?.id, crop: c, opts: o }));
    setView("bricks");
  };

  const signature = JSON.stringify({ id: photo?.id, crop, opts });
  const stale = !!matrix && generatedWith !== signature;

  const reqId = useRef(0);
  const [updating, setUpdating] = useState(false);

  // auto = a live update after a settings change: no blocking overlay, latest request wins.
  const generate = async (auto = false) => {
    if (!photo || !photoBlob) return;
    const id = ++reqId.current;
    setError("");
    if (auto) setUpdating(true);
    else setBusy(`Converting to ${opts.width}×${opts.height} bricks…`);
    try {
      const m =
        opts.engine === "studio" && image
          ? await api.convertStudio(image, photo.width, crop, framePhoto?.faces ?? photo.faces, opts)
          : await api.convert(photoBlob, crop, opts, job);
      if (id !== reqId.current) return;
      setMatrix(m);
      setGrid(m.grid);
      setPast([]);
      setFuture([]);
      setGeneratedWith(signature);
      setSelected(null);
      if (!auto || view === "sizes") generateAllSizes();
    } catch (e) {
      if (id === reqId.current) fail(e);
    } finally {
      if (id === reqId.current) {
        setBusy("");
        setUpdating(false);
      }
    }
  };

  const colorIndex = useMemo(() => {
    const map = new Map<string, PaletteColor>();
    for (const c of client) map.set(c.code, c);
    for (const c of matrix?.palette_full ?? []) map.set(c.code, c);
    return map;
  }, [client, matrix]);
  const hexOf = useCallback((code: string) => colorIndex.get(code)?.hex ?? "#ff00ff", [colorIndex]);
  const nameOf = useCallback((code: string) => colorIndex.get(code)?.name ?? "", [colorIndex]);

  const counts = useMemo(() => {
    const m = new Map<string, number>();
    for (const row of grid ?? []) for (const c of row) m.set(c, (m.get(c) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }, [grid]);
  const total = counts.reduce((s, [, q]) => s + q, 0);
  const countOf = useMemo(() => new Map(counts), [counts]);

  const startStroke = () => {
    if (!grid) return;
    setPast((p) => [...p.slice(-HISTORY + 1), grid]);
    setFuture([]);
  };
  const paint = (r: number, c: number) => {
    if (!selected) return;
    setGrid((g) => {
      if (!g || g[r][c] === selected) return g;
      const next = g.slice();
      next[r] = g[r].slice();
      next[r][c] = selected;
      return next;
    });
  };
  const undo = () => {
    if (!past.length || !grid) return;
    setFuture((f) => [grid, ...f]);
    setGrid(past[past.length - 1]);
    setPast((p) => p.slice(0, -1));
  };
  const redo = () => {
    if (!future.length || !grid) return;
    setPast((p) => [...p, grid]);
    setGrid(future[0]);
    setFuture((f) => f.slice(1));
  };

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || (e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key.toLowerCase() === "z" && !e.shiftKey) (e.preventDefault(), undo());
      if (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey)) (e.preventDefault(), redo());
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });

  const edits = useMemo(() => {
    if (!grid || !matrix) return 0;
    let k = 0;
    grid.forEach((row, r) => row.forEach((c, i) => c !== matrix.grid[r][i] && k++));
    return k;
  }, [grid, matrix]);

  // Live updates like a slider editor: regenerate shortly after the last change. Skipped once
  // bricks have been painted, so edits are never silently replaced (the button still works).
  useEffect(() => {
    if (!matrix || !stale || edits > 0) return;
    const t = setTimeout(() => generate(true), 600);
    return () => clearTimeout(t);
  }, [signature]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Remove a colour from the mosaic: its bricks move to the nearest colour still in use,
   * matched against each brick's target colour (same CIEDE2000 as the engine). Undoable. */
  const removeColour = (code: string) => {
    if (!grid || !matrix?.source) return;
    const kL = Number(matrix.settings.kL ?? 1);
    const allowed = counts.filter(([c]) => c !== code).map(([c]) => ({ code: c, lab: hexToLab(hexOf(c)) }));
    if (!allowed.length) return;
    startStroke();
    const src = matrix.source;
    setGrid((g) => g && g.map((row, r) => (row.includes(code) ? row.map((c, i) => (c === code ? nearest(src[r].slice(i * 6, i * 6 + 6), allowed, kL) : c)) : row)));
    if (selected === code) setSelected(null);
  };

  const [print, setPrint] = useState(PRINT_DEFAULTS);
  const setP = <K extends keyof typeof PRINT_DEFAULTS>(k: K, v: (typeof PRINT_DEFAULTS)[K]) => setPrint((o) => ({ ...o, [k]: v }));

  const downloadPrint = async () => {
    if (!matrix || !image || !photo) return;
    setBusy("Building the print file…");
    try {
      const [cols, rows] = dimsOf(matrix);
      const o = { ...print, cols, rows };
      const blob = await renderPrintSheet(image, photo.width, (matrix.settings.crop_resolved as Crop) ?? crop, o);
      const { dpi } = sheetInfo(o);
      download(blob, `print_${cols}x${rows}_${o.layout}_${o.tileMm}mm_${dpi}dpi.png`);
    } catch (e) {
      fail(e);
    } finally {
      setBusy("");
    }
  };

  const exportZip = async () => {
    if (!matrix || !grid) return;
    setBusy("Building download pack…");
    try {
      const blob = await api.exportPack({ ...matrix, grid, edited: edits > 0 });
      download(blob, `mosaic_${dimsOf(matrix).join("x")}${edits ? "_edited" : ""}.zip`);
    } catch (e) {
      fail(e);
    } finally {
      setBusy("");
    }
  };

  const set = <K extends keyof ConvertOptions>(k: K, v: ConvertOptions[K]) => setOpts((o) => ({ ...o, [k]: v }));

  // Dimensions of the generated mosaic (may differ from the settings until it regenerates).
  const [mw, mh] = matrix ? dimsOf(matrix) : [opts.width, opts.height];

  /** Board in plates; the crop follows the board's shape (same centre, short side kept). */
  const setBoard = (platesW: number, platesH: number) => {
    const width = platesW * PLATE;
    const height = platesH * PLATE;
    setOpts((o) => ({ ...o, width, height }));
    if (!photo) return;
    if (opts.engine === "studio" && autoCrop.current && framePhoto) {
      setCrop(studioCrop(framePhoto, width, height, opts.studio_auto_frame));
      return;
    }
    // Reshape from the crop the user last chose, not the current one: a tall board may have to
    // shrink the crop to fit the photo, and that must not carry over to the next shape.
    setCrop((c) => reshapeCrop(userCrop.current ?? c, photo.width, photo.height, width / height));
  };

  /** Studio "Auto (faces)": the reference framing for the current board. */
  const studioAuto = (o = opts, faces = framePhoto?.faces) => {
    if (!photo) return;
    autoCrop.current = true;
    setCrop(studioCrop({ ...photo, faces: faces ?? photo.faces }, o.width, o.height, o.studio_auto_frame));
  };
  const setEngine = async (engine: ConvertOptions["engine"]) => {
    setOpts((o) => ({ ...o, engine }));
    if (engine !== "studio" || !photo) return;
    const faces = studioFaces ?? (image ? await findStudioFaces(image, photo) : photo.faces);
    if (autoCrop.current) studioAuto({ ...opts, engine }, faces);
  };

  type StudioKey = "studio_colors" | "studio_contrast" | "studio_saturation" | "studio_smoothing" | "studio_local_contrast"
    | "studio_dark_lines" | "studio_detail" | "studio_dither" | "studio_bg_simplify";
  const studioSlider = (label: string, k: StudioKey, min: number, max: number, step: number) => (
    <label className="slider adj" key={k}>
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={opts[k]} onChange={(e) => set(k, +e.target.value)} />
      <output>
        {step < 1 ? opts[k].toFixed(2).replace(/0$/, "") : opts[k]}
        {opts[k] !== DEFAULTS[k] && (
          <button className="reset" title="Reset" onClick={(e) => (e.preventDefault(), set(k, DEFAULTS[k]))}>
            ↺
          </button>
        )}
      </output>
    </label>
  );
  type StudioFlag = "studio_auto_frame" | "studio_brighten_face" | "studio_auto_levels" | "studio_auto_saturation";
  const studioCheck = (label: string, k: StudioFlag) => (
    <label className="check" key={k}>
      <input
        type="checkbox"
        checked={opts[k]}
        onChange={(e) => {
          const o = { ...opts, [k]: e.target.checked };
          setOpts(o);
          if (k === "studio_auto_frame" && autoCrop.current) studioAuto(o);
        }}
      />{" "}
      {label}
    </label>
  );

  type NumKey = "detail" | "face_detail" | "dither" | "brightness" | "contrast" | "saturation" | "highlights";
  const adjustSlider = (label: string, k: NumKey, min: number, max: number, step: number, def: number) => (
    <label className="slider adj" key={k}>
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={opts[k]} onChange={(e) => set(k, +e.target.value)} />
      <output>
        {step < 1 ? opts[k].toFixed(step < 0.1 ? 2 : 1) : opts[k]}
        {opts[k] !== def && (
          <button className="reset" title="Reset" onClick={(e) => (e.preventDefault(), set(k, def))}>
            ↺
          </button>
        )}
      </output>
    </label>
  );

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>
            <i />
            <i />
            <i />
            <i />
          </span>
          <div>
            <h1>PIECES &amp; BRICKS</h1>
            <p>Mosaic Studio · demo</p>
          </div>
        </div>
        {matrix && (
          <div className="top-stats">
            <span>
              <b>
                {cm(mw)} × {cm(mh)} cm
              </b>{" "}
              · {mw}×{mh}
              {mw % PLATE === 0 && mh % PLATE === 0 && ` · ${mw / PLATE}×${mh / PLATE} plates`}
            </span>
            <span>
              <b>{total.toLocaleString()}</b> bricks
            </span>
            <span>
              <b>{counts.length}</b> colours
            </span>
          </div>
        )}
      </header>

      {server !== "ready" && (
        <div className={`banner ${server}`} role="status">
          {server === "waking" ? (
            <>
              <div className="spinner sm" /> Starting the mosaic server… the first visit after a quiet period can take up to a minute.
            </>
          ) : (
            <>The mosaic server is not answering. Please try again in a minute.</>
          )}
        </div>
      )}

      {needPw && (
        <div className="modal-back">
          <form
            className="modal"
            onSubmit={(e) => {
              e.preventDefault();
              api.setPassword(pwInput);
              setNeedPw(false);
              setError("");
              loadBasics();
            }}
          >
            <h3>Demo password</h3>
            <p className="muted">This demo is private. Enter the password you were given.</p>
            <input type="password" autoComplete="current-password" autoFocus value={pwInput} onChange={(e) => setPwInput(e.target.value)} aria-label="Password" />
            <button className="btn primary block" type="submit" disabled={!pwInput}>
              Continue
            </button>
          </form>
        </div>
      )}

      <main className="layout">
        {/* ---------- left: photo, crop, settings ---------- */}
        <aside className="panel left">
          <section>
            <h2>
              <span className="step">1</span> Photo
            </h2>
            <div
              className="drop"
              onClick={() => fileInput.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files[0];
                if (f) openPhoto(async () => ({ blob: f }));
              }}
            >
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) openPhoto(async () => ({ blob: f }));
                  e.target.value = "";
                }}
              />
              <b>Upload a photo</b>
              <span>Drop it here or click to browse · JPG, PNG up to 25 MB</span>
            </div>
            {samples.length > 0 && (
              <div className="samples">
                <span className="muted">Or try a client photo:</span>
                <div className="chips">
                  {samples.map((s) => (
                    <button key={s.key} className="chip" onClick={() => openPhoto(async () => ({ blob: await api.sampleFile(s.key), job: s.job }))}>
                      {s.title}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </section>

          {photo && (
            <section>
              <h2>
                <span className="step">2</span> Crop &amp; zoom
              </h2>
              {opts.engine === "studio" && facesNote && <p className="hint">{facesNote}</p>}
              <CropEditor
                photo={framePhoto ?? photo}
                image={image}
                crop={crop}
                cols={opts.width}
                rows={opts.height}
                onChange={(c) => {
                  userCrop.current = c;
                  autoCrop.current = false;
                  setCrop(c);
                }}
                onAuto={opts.engine === "studio" ? () => studioAuto() : undefined}
              />
            </section>
          )}

          {photo && (
            <section>
              <h2>
                <span className="step">3</span> Settings
              </h2>
              <div className="field">
                <span className="label">Board size</span>
                <div className="seg small shape-tabs">
                  {(
                    [
                      ["square", "Square"],
                      ["landscape", "Landscape"],
                      ["portrait", "Portrait"],
                      ["custom", "Custom"],
                    ] as [Shape, string][]
                  ).map(([v, label]) => (
                    <button key={v} className={shape === v ? "on" : ""} onClick={() => setShape(v)}>
                      {label}
                    </button>
                  ))}
                </div>
                {shape === "custom" ? (
                  <div className="custom-board">
                    <label>
                      Plates wide
                      <input type="number" min={1} max={MAX_PLATES} value={opts.width / PLATE}
                        onChange={(e) => setBoard(Math.min(MAX_PLATES, Math.max(1, Math.round(+e.target.value) || 1)), opts.height / PLATE)} />
                    </label>
                    <span>×</span>
                    <label>
                      Plates high
                      <input type="number" min={1} max={MAX_PLATES} value={opts.height / PLATE}
                        onChange={(e) => setBoard(opts.width / PLATE, Math.min(MAX_PLATES, Math.max(1, Math.round(+e.target.value) || 1)))} />
                    </label>
                  </div>
                ) : (
                  <div className="sizes">
                    {BOARDS[shape].map(([pw, ph]) => {
                      const on = opts.width === pw * PLATE && opts.height === ph * PLATE;
                      return (
                        <button key={`${pw}x${ph}`} className={`size-card ${on ? "on" : ""}`} onClick={() => setBoard(pw, ph)}>
                          <span className="plates" style={{ gridTemplateColumns: `repeat(${pw}, 1fr)`, aspectRatio: `${pw} / ${ph}` }} aria-hidden>
                            {Array.from({ length: pw * ph }).map((_, i) => (
                              <i key={i} />
                            ))}
                          </span>
                          <b>
                            {cm(pw * PLATE)} × {cm(ph * PLATE)}
                          </b>
                          <span className="muted">
                            {pw}×{ph} plates · {pw * PLATE}×{ph * PLATE}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}
                <p className="hint">
                  {opts.width / PLATE}×{opts.height / PLATE} plates of 24×24 studs = {cm(opts.width)} × {cm(opts.height)} cm ·{" "}
                  {(opts.width * opts.height).toLocaleString()} bricks · frame opening {frameCm(opts.width)} × {frameCm(opts.height)} cm
                </p>
              </div>
              <div className="field">
                <span className="label">Engine</span>
                <div className="seg">
                  <button className={opts.engine === "studio" ? "on" : ""} onClick={() => setEngine("studio")}>
                    Studio
                  </button>
                  <button className={opts.engine === "classic" ? "on" : ""} onClick={() => setEngine("classic")}>
                    Classic
                  </button>
                </div>
              </div>

              {opts.engine === "studio" ? (
                <>
                  <div className="field">
                    <span className="label">Palette</span>
                    <div className="seg">
                      <button className={opts.studio_palette === "40" ? "on" : ""} onClick={() => set("studio_palette", "40")}>
                        40 colours
                      </button>
                      <button className={opts.studio_palette === "current" ? "on" : ""} onClick={() => set("studio_palette", "current")}>
                        {client.length || 59} colours
                      </button>
                    </div>
                  </div>
                  <div className="adjust">
                    {studioSlider("Colors", "studio_colors", 6, opts.studio_palette === "40" ? 40 : client.length || 59, 1)}
                    {studioSlider("Contrast", "studio_contrast", -30, 60, 1)}
                    {opts.studio_auto_saturation ? (
                      <label className="slider adj">
                        <span>Saturation</span>
                        <span className="muted">
                          Auto{typeof matrix?.settings.studio_saturation_used === "number" ? `: ${matrix.settings.studio_saturation_used}` : ""}
                        </span>
                        <span />
                      </label>
                    ) : (
                      studioSlider("Saturation", "studio_saturation", -30, 60, 1)
                    )}
                    {studioSlider("Smoothing", "studio_smoothing", 0, 1, 0.05)}
                    {studioSlider("Local contrast", "studio_local_contrast", 0, 1.5, 0.05)}
                    {studioSlider("Dark lines", "studio_dark_lines", 0, 2, 0.05)}
                    {studioSlider("Detail", "studio_detail", 0, 0.6, 0.02)}
                    {studioSlider("Dithering", "studio_dither", 0, 1, 0.05)}
                    {studioSlider("Background simplify", "studio_bg_simplify", 0, 1, 0.05)}
                  </div>
                  <div className="checks">
                    {studioCheck("Auto-frame faces (tighter on small sizes)", "studio_auto_frame")}
                    {studioCheck("Brighten face", "studio_brighten_face")}
                    {studioCheck("Auto levels", "studio_auto_levels")}
                    {studioCheck("Auto saturation (from the photo's colour)", "studio_auto_saturation")}
                  </div>
                </>
              ) : (
              <>
              <div className="field">
                <span className="label">Colours</span>
                <label className="slider adj">
                  <span>{opts.colors >= (client.length || 59) ? "All" : `Best ${opts.colors}`}</span>
                  <input type="range" min={6} max={client.length || 59} value={opts.colors} onChange={(e) => set("colors", +e.target.value)} />
                  <output>
                    {opts.colors}
                    {opts.colors !== DEFAULTS.colors && (
                      <button className="reset" title="Auto (20)" onClick={(e) => (e.preventDefault(), set("colors", DEFAULTS.colors))}>
                        ↺
                      </button>
                    )}
                  </output>
                </label>
                <p className="hint">Picks the best {Math.min(opts.colors, client.length || 59)} of the {client.length || 59} brick colours for this photo. 20 gives clean, even areas.</p>
              </div>
              <div className="field">
                <span className="label">Blend colours</span>
                <div className="seg">
                  {(
                    [
                      ["off", "Off"],
                      ["background", "Textured areas"],
                      ["all", "Everywhere"],
                    ] as const
                  ).map(([v, label]) => (
                    <button key={v} className={opts.blend === v ? "on" : ""} onClick={() => set("blend", v)}>
                      {label}
                    </button>
                  ))}
                </div>
                <p className="hint">
                  Off: every brick is its own closest colour (cleanest, best for photos). Textured areas: neighbouring bricks mix like
                  paint where the picture has texture, faces stay clean (paintings, art).
                </p>
              </div>
              <div className="field">
                <span className="label">Background</span>
                <div className="seg">
                  {(
                    [
                      ["bilateral", "Smooth"],
                      ["meanshift", "Flat"],
                      ["none", "Original"],
                    ] as const
                  ).map(([v, t]) => (
                    <button key={v} className={opts.background === v ? "on" : ""} onClick={() => set("background", v)}>
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              <span className="label">Adjust</span>
              <div className="adjust">
                {adjustSlider("Detail", "detail", 0, 5, 0.5, DEFAULTS.detail)}
                {adjustSlider("Face detail", "face_detail", 0, 1.5, 0.05, DEFAULTS.face_detail)}
                {adjustSlider("Dither", "dither", 0, 100, 5, 0)}
                {adjustSlider("Brightness", "brightness", -100, 100, 1, DEFAULTS.brightness)}
                {adjustSlider("Contrast", "contrast", -100, 100, 1, DEFAULTS.contrast)}
                {adjustSlider("Saturation", "saturation", -100, 100, 1, 0)}
                {adjustSlider("Highlights", "highlights", 0, 100, 1, 0)}
              </div>

              <div className="field">
                <span className="label">Filter</span>
                <div className="chips">
                  {FILTERS.map(([v, t]) => (
                    <button key={v} className={`chip ${opts.filter === v ? "on" : ""}`} onClick={() => set("filter", v)}>
                      {t}
                    </button>
                  ))}
                </div>
              </div>
              </>
              )}

              <button className="btn primary block" onClick={() => generate()} disabled={!!busy}>
                {matrix ? (stale ? "Generate again" : "Regenerate") : "Generate mosaic"}
              </button>
              {stale && edits > 0 && (
                <p className="hint warn">Settings changed. Generate again to apply them (this replaces your {edits} painted bricks).</p>
              )}
              {matrix && !(stale && edits > 0) && <p className="hint">Changes update the mosaic automatically.</p>}
            </section>
          )}
        </aside>

        {/* ---------- centre: mosaic ---------- */}
        <section className="stage">
          {grid && matrix ? (
            <>
              <div className="toolbar">
                <div className="seg">
                  {(
                    [
                      ["sizes", "All sizes"],
                      ["bricks", "Bricks"],
                      ["flat", "Flat"],
                      ["numbers", "Numbered grid"],
                      ["compare", "Compare"],
                      ["printed", "Printed tiles"],
                    ] as const
                  ).map(([v, t]) => (
                    <button key={v} className={view === v ? "on" : ""} onClick={() => setView(v)}>
                      {t}
                    </button>
                  ))}
                </div>
                {updating && (
                  <span className="updating" role="status">
                    <span className="spinner sm" /> Updating…
                  </span>
                )}
                <div className="tools" hidden={view === "printed" || view === "sizes"}>
                  <div className="seg">
                    <button className={tool === "paint" ? "on" : ""} onClick={() => setTool("paint")} title="Paint bricks with the selected colour">
                      Paint
                    </button>
                    <button className={tool === "pick" ? "on" : ""} onClick={() => setTool("pick")} title="Pick a colour from the mosaic">
                      Pick
                    </button>
                  </div>
                  <button className="btn ghost sm" onClick={undo} disabled={!past.length} title="Undo (Ctrl+Z)">
                    Undo
                  </button>
                  <button className="btn ghost sm" onClick={redo} disabled={!future.length} title="Redo (Ctrl+Y)">
                    Redo
                  </button>
                  <button
                    className="btn ghost sm"
                    disabled={!edits}
                    onClick={() => {
                      startStroke();
                      setGrid(matrix.grid);
                    }}
                  >
                    Reset
                  </button>
                </div>
              </div>
              {view === "sizes" ? (
                <SizeGrid
                  results={sizeResults}
                  current={[mw, mh]}
                  plate={PLATE}
                  cm={cm}
                  hexOf={hexOf}
                  onOpen={setBigSize}
                />
              ) : (
              <MosaicCanvas
                grid={grid}
                hexOf={hexOf}
                nameOf={nameOf}
                view={view}
                tool={tool}
                image={image}
                photoWidth={photo?.width ?? 1}
                crop={(matrix.settings.crop_resolved as Crop) ?? crop}
                onStrokeStart={() => selected && startStroke()}
                onPaint={paint}
                onPick={(c) => {
                  setSelected(c);
                  setTool("paint");
                }}
              />
              )}
              {tool === "paint" && !selected && view !== "compare" && view !== "printed" && view !== "sizes" && (
                <p className="hint center">Choose a colour on the right, then click or drag on the mosaic to paint.</p>
              )}
            </>
          ) : (
            <div className="empty">
              <div className="empty-art" aria-hidden>
                {Array.from({ length: 36 }).map((_, i) => (
                  <i key={i} style={{ background: ["#FD637D", "#FFD300", "#58A3DA", "#ECA279", "#00A64D", "#F7F0E3"][(i * 7) % 6] }} />
                ))}
              </div>
              <h3>{photo ? "Set the crop, then generate" : "Start with a photo"}</h3>
              <p className="muted">
                {photo
                  ? "Frame the faces large: the bigger they are in the crop, the clearer they look in bricks."
                  : "Upload a portrait, or try one of the client photos."}
              </p>
            </div>
          )}
          {busy && (
            <div className="busy" role="status">
              <div className="spinner" />
              {busy}
            </div>
          )}
          {error && (
            <div className="error" role="alert">
              {error} <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
        </section>

        {/* ---------- right: colours + counts ---------- */}
        {grid && matrix && (
          <aside className="panel right">
            <section>
              <h2>Paint colour</h2>
              <div className="seg small">
                <button className="on">Color {client.length || 59}</button>
              </div>
              <div className="swatches">
                {client.map((c) => (
                  <button
                    key={c.code}
                    className={`swatch ${selected === c.code ? "on" : ""}`}
                    style={{ background: c.hex }}
                    title={`${c.code} ${c.name} ${c.hex}`}
                    onClick={() => {
                      setSelected(c.code);
                      setTool("paint");
                    }}
                  />
                ))}
              </div>
              {selected && (
                <div className="selected">
                  <span className="sw lg" style={{ background: hexOf(selected) }} />
                  <div>
                    <b>{selected}</b> {nameOf(selected)}
                    <div className="muted mono">
                      {hexOf(selected)} · {countOf.get(selected) ?? 0} bricks
                    </div>
                  </div>
                </div>
              )}
            </section>

            <section className="counts">
              <h2>
                Brick counts <span className="muted">· {counts.length} colours{edits ? ` · ${edits} edited` : ""}</span>
              </h2>
              <div className="table">
                <div className="tr th">
                  <span>Code</span>
                  <span>Name</span>
                  <span className="num">Qty</span>
                  <span />
                </div>
                {counts.map(([code, q]) => (
                  <div key={code} role="button" tabIndex={0} className={`tr ${selected === code ? "on" : ""}`} onClick={() => setSelected(code)}
                    onKeyDown={(e) => e.key === "Enter" && setSelected(code)}>
                    <span className="code">
                      <span className="sw" style={{ background: hexOf(code) }} />
                      {code}
                    </span>
                    <span className="name">{nameOf(code)}</span>
                    <span className="num">{q}</span>
                    <button
                      className="rm"
                      title={`Remove ${code}: its bricks switch to the nearest colour in use`}
                      disabled={counts.length <= 1 || !matrix.source}
                      onClick={(e) => {
                        e.stopPropagation();
                        removeColour(code);
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
                <div className={`tr total ${total === mw * mh ? "ok" : "bad"}`}>
                  <span>Total</span>
                  <span>{total === mw * mh ? "matches the board ✓" : "does not match the board"}</span>
                  <span className="num">{total.toLocaleString()}</span>
                  <span />
                </div>
              </div>
            </section>

            <section>
              <button className="btn primary block" onClick={exportZip} disabled={!!busy}>
                Download pack (.zip)
              </button>
              <p className="hint">preview.png, grid.png, flat.png, counts.csv and matrix.json, all rendered from the current grid, including your edits.</p>
            </section>

            <section className="print-panel">
              <h2>
                Printed tiles <span className="tag">optional</span>
              </h2>
              <p className="hint intro">
                Instead of coloured bricks, UV-print the photo itself onto {(mw * mh).toLocaleString()} blank flat 1×1 tiles.
                See it in the <button className="link" onClick={() => setView("printed")}>Printed tiles</button> view.
              </p>
              <div className="field">
                <span className="label">Layout</span>
                <div className="seg small">
                  {(
                    [
                      ["seamless", "One sheet"],
                      ["jig", "Tile jig"],
                    ] as [PrintLayout, string][]
                  ).map(([v, t]) => (
                    <button key={v} className={print.layout === v ? "on" : ""} onClick={() => setP("layout", v)}>
                      {t}
                    </button>
                  ))}
                </div>
              </div>
              <div className="print-grid">
                <label>
                  Tile size
                  <span>
                    <input type="number" min={2} max={50} step={0.1} value={print.tileMm} onChange={(e) => setP("tileMm", +e.target.value || 8)} /> mm
                  </span>
                </label>
                <label>
                  Resolution
                  <span>
                    <input type="number" min={72} max={1200} step={10} value={print.dpi} onChange={(e) => setP("dpi", +e.target.value || 300)} /> dpi
                  </span>
                </label>
                {print.layout === "jig" && (
                  <>
                    <label>
                      Jig gap
                      <span>
                        <input type="number" min={0} max={20} step={0.1} value={print.gapMm} onChange={(e) => setP("gapMm", +e.target.value || 0)} /> mm
                      </span>
                    </label>
                    <label>
                      Bleed
                      <span>
                        <input type="number" min={0} max={2} step={0.05} value={print.bleedMm} onChange={(e) => setP("bleedMm", +e.target.value || 0)} /> mm
                      </span>
                    </label>
                  </>
                )}
              </div>
              {(() => {
                const info = sheetInfo({ ...print, cols: mw, rows: mh });
                return (
                  <p className="hint">
                    Sheet {(info.widthMm / 10).toFixed(1)} × {(info.heightMm / 10).toFixed(1)} cm · {info.widthPx.toLocaleString()} ×{" "}
                    {info.heightPx.toLocaleString()} px at {info.dpi} dpi
                    {info.dpi < print.dpi && " (lowered: the browser can't make a bigger image)"}.
                    {print.layout === "jig"
                      ? " Tiles are spaced as they sit in a printer jig; the photo runs a little past each edge (bleed)."
                      : " One image the size of the finished board, for printing on a plate or a full sheet of tiles."}
                  </p>
                );
              })()}
              <button className="btn block" onClick={downloadPrint} disabled={!!busy || !image}>
                Download print file (.png)
              </button>
              <p className="hint">Made from the full-resolution photo, not the brick colours. Pieces needed: {(mw * mh).toLocaleString()} white flat 1×1 tiles.</p>
            </section>
          </aside>
        )}
      </main>
      {bigSize?.matrix && (() => {
        const [pw, ph] = bigSize.plates;
        const title = `${cm(pw * PLATE)} × ${cm(ph * PLATE)} cm · ${pw}×${ph} baseplates, ${pw * PLATE}×${ph * PLATE} studs`;
        return (
          <BigView
            result={bigSize}
            title={title}
            hexOf={hexOf}
            nameOf={nameOf}
            onClose={() => setBigSize(null)}
            onEdit={() => {
              openSize(bigSize);
              setBigSize(null);
            }}
            onDownload={async () => download(await posterPng(bigSize.matrix!, hexOf, nameOf, title), `brick-mosaic-${pw}x${ph}.png`)}
          />
        );
      })()}
    </div>
  );
}
