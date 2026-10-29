import { useEffect, useRef, useState } from "react";
import type { Crop } from "../api";
import { brickTile, luminance } from "../bricks";

export type View = "bricks" | "flat" | "numbers" | "compare";
export type Tool = "paint" | "pick";

interface Props {
  grid: string[][];
  hexOf: (code: string) => string;
  nameOf: (code: string) => string;
  view: View;
  tool: Tool;
  image: HTMLImageElement | null;
  photoWidth: number;
  crop: Crop;
  onStrokeStart: () => void;
  onPaint: (row: number, col: number) => void;
  onPick: (code: string) => void;
}

const BLOCK = 8;
const ZOOMS = [1, 2, 4, 8];
const MAX_CANVAS = 8192; // device pixels per side; keeps big boards within browser limits

export default function MosaicCanvas(p: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const painting = useRef(false);
  const last = useRef<string>("");
  const [width, setWidth] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [split, setSplit] = useState(0.5);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const n = p.grid.length;
  const labels = p.view === "numbers";

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Numbered grids of big boards need room for the digits: start them zoomed in.
  useEffect(() => {
    if (labels && n > 64 && zoom === 1) setZoom(n > 128 ? 4 : 2);
  }, [labels, n]); // zoom deliberately left out: only react to view / size changes

  // Layout: in numbers view keep a margin for row/column labels.
  const dpr = window.devicePixelRatio || 1;
  const margin = labels ? 1 : 0;
  const cells = n + 2 * margin;
  const cell = Math.max(2, Math.min(Math.floor((width * zoom * dpr) / cells), Math.floor(MAX_CANVAS / cells)));
  const off = margin * cell;
  const W = cell * cells;
  // At 1x fill the panel exactly (bricks rarely divide the width evenly); zoomed, use true size.
  const cssW = zoom === 1 && width ? width : W / dpr;

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !width) return;
    cv.width = cv.height = W;
    const ctx = cv.getContext("2d")!;
    ctx.fillStyle = labels ? "#fff" : "rgb(24,22,26)";
    ctx.fillRect(0, 0, W, W);

    const drawCells = (mode: View) => {
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const hex = p.hexOf(p.grid[r][c]);
          const x = off + c * cell;
          const y = off + r * cell;
          if ((mode === "bricks" || mode === "compare") && cell >= 4) ctx.drawImage(brickTile(hex, cell), x, y);
          else {
            ctx.fillStyle = hex;
            ctx.fillRect(x, y, cell, cell);
          }
        }
      }
    };

    if (p.view === "numbers") {
      drawCells("flat");
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const wide = p.grid.some((row) => row.some((code) => code.length > 3));
      if (cell >= (wide ? 14 : 11)) {
        ctx.font = `600 ${Math.floor(cell * (wide ? 0.32 : 0.42))}px system-ui, sans-serif`;
        for (let r = 0; r < n; r++)
          for (let c = 0; c < n; c++) {
            const code = p.grid[r][c];
            ctx.fillStyle = luminance(p.hexOf(code)) > 140 ? "#000" : "#fff";
            ctx.fillText(code.slice(1), off + c * cell + cell / 2, off + r * cell + cell / 2 + 0.5);
          }
      }
      for (let i = 0; i <= n; i++) {
        const thick = i % BLOCK === 0;
        if (!thick && cell < 6) continue;
        ctx.strokeStyle = thick ? "#000" : "rgba(60,60,60,0.7)";
        ctx.lineWidth = thick ? Math.max(2, dpr * 1.5) : 1;
        const q = off + i * cell;
        ctx.beginPath();
        ctx.moveTo(q, off);
        ctx.lineTo(q, off + n * cell);
        ctx.moveTo(off, q);
        ctx.lineTo(off + n * cell, q);
        ctx.stroke();
      }
      ctx.fillStyle = "#333";
      ctx.font = `${Math.floor(cell * 0.38)}px system-ui, sans-serif`;
      const every = cell >= 14 ? 1 : cell >= 7 ? 2 : BLOCK;
      for (let i = 0; i < n; i++) {
        if ((i + 1) % every && i) continue;
        const q = off + i * cell + cell / 2;
        ctx.fillText(String(i + 1), q, off / 2);
        ctx.fillText(String(i + 1), off / 2, q);
      }
    } else {
      drawCells(p.view);
    }

    if (p.view === "compare" && p.image) {
      const vs = p.image.naturalWidth / p.photoWidth;
      const sx = Math.round(W * split);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, sx, W);
      ctx.clip();
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(p.image, p.crop[0] * vs, p.crop[1] * vs, p.crop[2] * vs, p.crop[2] * vs, 0, 0, W, W);
      ctx.restore();
      ctx.fillStyle = "#fff";
      ctx.fillRect(sx - dpr, 0, 2 * dpr, W);
    }
  }, [p.grid, p.view, p.image, p.crop, p.photoWidth, p.hexOf, width, split, cell, off, W, n, dpr, labels]);

  // Hover highlight lives on its own layer so big boards are not redrawn on every mouse move.
  useEffect(() => {
    const cv = overlay.current;
    if (!cv) return;
    if (cv.width !== W) cv.width = cv.height = W;
    const ctx = cv.getContext("2d")!;
    ctx.clearRect(0, 0, W, W);
    if (!hover || p.view === "compare") return;
    const x = off + hover.c * cell;
    const y = off + hover.r * cell;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = Math.max(1, dpr);
    ctx.strokeRect(x - 1, y - 1, cell + 2, cell + 2);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = Math.max(1.5, dpr * 1.5);
    ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
  }, [hover, W, off, cell, dpr, p.view]);

  const cellAt = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const k = W / rect.width;
    const c = Math.floor(((e.clientX - rect.left) * k - off) / cell);
    const r = Math.floor(((e.clientY - rect.top) * k - off) / cell);
    return r >= 0 && c >= 0 && r < n && c < n ? { r, c } : null;
  };

  const apply = (at: { r: number; c: number }) => {
    if (p.tool === "pick") return p.onPick(p.grid[at.r][at.c]);
    const key = `${at.r},${at.c}`;
    if (key === last.current) return;
    last.current = key;
    p.onPaint(at.r, at.c);
  };

  const editable = p.view !== "compare";
  const zoomTo = (z: number) => {
    const el = wrap.current;
    if (!el) return setZoom(z);
    // keep the centre of the view in place
    const cx = (el.scrollLeft + el.clientWidth / 2) / (el.scrollWidth || 1);
    const cy = (el.scrollTop + el.clientHeight / 2) / (el.scrollHeight || 1);
    setZoom(z);
    requestAnimationFrame(() => {
      el.scrollLeft = cx * el.scrollWidth - el.clientWidth / 2;
      el.scrollTop = cy * el.scrollHeight - el.clientHeight / 2;
    });
  };

  return (
    <div className="mosaic">
      <div className={`mosaic-wrap ${zoom > 1 ? "zoomed" : ""}`} ref={wrap}>
        <div className="mosaic-layers" style={{ width: cssW || "100%", height: cssW || "auto" }}>
          <canvas ref={ref} className="mosaic-canvas" style={{ width: cssW, height: cssW }} />
          <canvas
            ref={overlay}
            className={`mosaic-overlay ${editable ? (p.tool === "pick" ? "pick" : "paint") : ""}`}
            style={{ width: cssW, height: cssW }}
            onPointerDown={(e) => {
              if (!editable) return;
              const at = cellAt(e);
              if (!at) return;
              e.currentTarget.setPointerCapture(e.pointerId);
              last.current = "";
              if (p.tool === "paint") {
                painting.current = true;
                p.onStrokeStart();
              }
              apply(at);
            }}
            onPointerMove={(e) => {
              const at = cellAt(e);
              setHover((h) => (h?.r === at?.r && h?.c === at?.c ? h : at));
              if (painting.current && at) apply(at);
            }}
            onPointerUp={() => (painting.current = false)}
            onPointerLeave={() => setHover(null)}
          />
        </div>
      </div>
      <div className="mosaic-foot">
        <div className="hover-info" aria-live="polite">
          {hover ? (
            <>
              <span className="sw" style={{ background: p.hexOf(p.grid[hover.r][hover.c]) }} />
              Row {hover.r + 1}, Col {hover.c + 1} · <b>{p.grid[hover.r][hover.c]}</b> {p.nameOf(p.grid[hover.r][hover.c])}
            </>
          ) : (
            <span className="muted">{editable ? "Hover a brick to inspect it. Click or drag to paint." : "Drag the slider to compare."}</span>
          )}
        </div>
        <div className="seg small" role="group" aria-label="Zoom">
          {ZOOMS.map((z) => (
            <button key={z} className={zoom === z ? "on" : ""} onClick={() => zoomTo(z)} title={`Zoom ${z}×`}>
              {z}×
            </button>
          ))}
        </div>
      </div>
      {p.view === "compare" && (
        <label className="slider split">
          <span>Original</span>
          <input type="range" min={0} max={1} step={0.005} value={split} onChange={(e) => setSplit(+e.target.value)} />
          <span>Bricks</span>
        </label>
      )}
    </div>
  );
}
