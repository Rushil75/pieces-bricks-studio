import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import type { ConvertOptions, Crop, Filter, Matrix, PaletteColor, Photo } from "./api";
import { hexToLab, nearest } from "./colour";
import CropEditor, { clampCrop } from "./components/CropEditor";
import MosaicCanvas, { type Tool, type View } from "./components/MosaicCanvas";
import { PRINT_DEFAULTS, renderPrintSheet, sheetInfo, type PrintLayout } from "./print";

const SIZES = [32, 48, 64, 96, 128, 160, 192, 256];
const STUD_MM = 8;
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
  size: 48,
  dither: 0,
  detail: 1,
  brightness: 0,
  contrast: 0,
  saturation: 0,
  filter: "none",
  palette: "adaptive",
  colors: 60,
  face_detail: 0.15,
  background: "bilateral",
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
  const [crop, setCrop] = useState<Crop>([0, 0, 1]);
  const [opts, setOpts] = useState<ConvertOptions>(DEFAULTS);
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [grid, setGrid] = useState<string[][] | null>(null);
  const [past, setPast] = useState<string[][][]>([]);
  const [future, setFuture] = useState<string[][][]>([]);
  const [generatedWith, setGeneratedWith] = useState<string>("");
  const [view, setView] = useState<View>("bricks");
  const [tool, setTool] = useState<Tool>("paint");
  const [selected, setSelected] = useState<string | null>(null);
  const [swatchTab, setSwatchTab] = useState<"mosaic" | "client">("mosaic");
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
      setCrop(clampCrop(ph.auto_crop, ph.width, ph.height));
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
    else setBusy(`Converting to ${opts.size}×${opts.size} bricks…`);
    try {
      const m = await api.convert(photoBlob, crop, opts, job);
      if (id !== reqId.current) return;
      setMatrix(m);
      setGrid(m.grid);
      setPast([]);
      setFuture([]);
      setGeneratedWith(signature);
      setSelected(null);
      setSwatchTab(m.palette.name === "client" ? "client" : "mosaic");
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
      const o = { ...print, tiles: matrix.size };
      const blob = await renderPrintSheet(image, photo.width, (matrix.settings.crop_resolved as Crop) ?? crop, o);
      const { dpi } = sheetInfo(o);
      download(blob, `print_${matrix.size}x${matrix.size}_${o.layout}_${o.tileMm}mm_${dpi}dpi.png`);
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
      download(blob, `mosaic_${matrix.size}x${matrix.size}${edits ? "_edited" : ""}.zip`);
    } catch (e) {
      fail(e);
    } finally {
      setBusy("");
    }
  };

  const swatches: PaletteColor[] =
    swatchTab === "client" ? client : [...(matrix?.palette_full ?? [])].sort((a, b) => (countOf.get(b.code) ?? 0) - (countOf.get(a.code) ?? 0));

  const set = <K extends keyof ConvertOptions>(k: K, v: ConvertOptions[K]) => setOpts((o) => ({ ...o, [k]: v }));

  type NumKey = "detail" | "face_detail" | "dither" | "brightness" | "contrast" | "saturation";
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
              <b>{matrix.size}×{matrix.size}</b> board · {((matrix.size * STUD_MM) / 10).toFixed(1)} × {((matrix.size * STUD_MM) / 10).toFixed(1)} cm
            </span>
            <span>
              <b>{total.toLocaleString()}</b> bricks
            </span>
            <span>
              <b>{counts.length}</b> colours
            </span>
            {matrix.similarity && (
              <span className={matrix.similarity.similarity >= 95 ? "sim-ok" : "sim-low"}>
                <b>{matrix.similarity.similarity.toFixed(1)}%</b> similar
              </span>
            )}
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
              <CropEditor photo={photo} image={image} crop={crop} gridSize={opts.size} onChange={setCrop} />
            </section>
          )}

          {photo && (
            <section>
              <h2>
                <span className="step">3</span> Settings
              </h2>
              <div className="field">
                <span className="label">Board size</span>
                <div className="seg">
                  {SIZES.map((s) => (
                    <button key={s} className={opts.size === s ? "on" : ""} onClick={() => set("size", s)}>
                      {s}×{s}
                    </button>
                  ))}
                </div>
              </div>
              <div className="field">
                <span className="label">Colours</span>
                <div className="seg">
                  <button className={opts.palette === "adaptive" ? "on" : ""} onClick={() => set("palette", "adaptive")}>
                    Best match
                  </button>
                  <button className={opts.palette === "client" ? "on" : ""} onClick={() => set("palette", "client")}>
                    Client 40
                  </button>
                </div>
              </div>
              {opts.palette === "adaptive" && (
                <label className="slider">
                  <span>Max colours</span>
                  <input type="range" min={16} max={60} value={opts.colors} onChange={(e) => set("colors", +e.target.value)} />
                  <output>{opts.colors}</output>
                </label>
              )}
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
                {adjustSlider("Brightness", "brightness", -100, 100, 1, 0)}
                {adjustSlider("Contrast", "contrast", -100, 100, 1, 0)}
                {adjustSlider("Saturation", "saturation", -100, 100, 1, 0)}
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
                <div className="tools" hidden={view === "printed"}>
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
              {tool === "paint" && !selected && view !== "compare" && view !== "printed" && (
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
            {matrix.similarity && (
              <section className="similarity">
                <h2>Similarity to photo</h2>
                <div className={`sim-big ${matrix.similarity.similarity >= 95 ? "ok" : "low"}`}>
                  {matrix.similarity.similarity.toFixed(1)}%
                </div>
                <div className="sim-bar">
                  <span style={{ width: `${Math.min(100, matrix.similarity.similarity)}%` }} />
                  <i style={{ left: "95%" }} title="95% target" />
                </div>
                <dl>
                  <dt>Bricks with a close colour match (ΔE ≤ 5)</dt>
                  <dd>{matrix.similarity.pct_close.toFixed(1)}%</dd>
                  <dt>Bricks with no visible difference (ΔE ≤ 2.3)</dt>
                  <dd>{matrix.similarity.pct_unnoticeable.toFixed(1)}%</dd>
                  <dt>Structure match (SSIM)</dt>
                  <dd>{matrix.similarity.structure_ssim.toFixed(1)}%</dd>
                  <dt>Average colour error (ΔE)</dt>
                  <dd>{matrix.similarity.mean_delta_e.toFixed(2)}</dd>
                  {matrix.similarity.face_mean_delta_e !== undefined && (
                    <>
                      <dt>Average colour error on faces</dt>
                      <dd>{matrix.similarity.face_mean_delta_e.toFixed(2)}</dd>
                    </>
                  )}
                </dl>
                <p className="hint">
                  Measured against the photo averaged under each brick, the most detail a {matrix.size}×{matrix.size} board can hold.
                  {edits > 0 && " Score is for the generated mosaic, before your edits."}
                  {(opts.filter !== "none" || opts.brightness || opts.contrast || opts.saturation || opts.detail > DEFAULTS.detail) &&
                    " Filters and adjustments change the picture on purpose, so they lower this score."}
                </p>
              </section>
            )}
            <section>
              <h2>Paint colour</h2>
              <div className="seg small">
                <button className={swatchTab === "mosaic" ? "on" : ""} onClick={() => setSwatchTab("mosaic")} disabled={matrix.palette.name === "client"}>
                  This mosaic ({matrix.palette_full.length})
                </button>
                <button className={swatchTab === "client" ? "on" : ""} onClick={() => setSwatchTab("client")}>
                  Client 40
                </button>
              </div>
              <div className="swatches">
                {swatches.map((c) => (
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
                <div className={`tr total ${total === matrix.size * matrix.size ? "ok" : "bad"}`}>
                  <span>Total</span>
                  <span>{total === matrix.size * matrix.size ? "matches the board ✓" : "does not match the board"}</span>
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
                Instead of coloured bricks, UV-print the photo itself onto {(matrix.size * matrix.size).toLocaleString()} blank flat 1×1 tiles.
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
                const info = sheetInfo({ ...print, tiles: matrix.size });
                return (
                  <p className="hint">
                    Sheet {(info.sideMm / 10).toFixed(1)} × {(info.sideMm / 10).toFixed(1)} cm · {info.sidePx.toLocaleString()} px at {info.dpi} dpi
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
              <p className="hint">Made from the full-resolution photo, not the brick colours. Pieces needed: {(matrix.size * matrix.size).toLocaleString()} white flat 1×1 tiles.</p>
            </section>
          </aside>
        )}
      </main>
    </div>
  );
}
