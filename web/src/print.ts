// Optional "printed tiles" mode: print-ready files for UV-printing the photo onto 1x1 tiles.
// Built in the browser from the full-resolution photo, so no server size limits apply.
import type { Crop } from "./api";

export type PrintLayout = "seamless" | "jig";

export interface PrintOptions {
  tiles: number; // tiles per side (the board size)
  tileMm: number; // tile pitch, 8 mm for standard 1x1 pieces
  dpi: number;
  layout: PrintLayout;
  gapMm: number; // jig only: space between tiles in the jig
  bleedMm: number; // jig only: image printed past each tile edge
}

export const PRINT_DEFAULTS: Omit<PrintOptions, "tiles"> = {
  tileMm: 8,
  dpi: 300,
  layout: "seamless",
  gapMm: 2,
  bleedMm: 0.3,
};

const MAX_SIDE = 16000; // browsers refuse larger canvases

const mmToPx = (mm: number, dpi: number) => (mm / 25.4) * dpi;

/** Sheet size in px and mm, and the DPI actually used (lowered if the sheet would be too big). */
export function sheetInfo(o: PrintOptions) {
  const sideMm = o.layout === "seamless" ? o.tiles * o.tileMm : o.tiles * o.tileMm + (o.tiles + 1) * o.gapMm;
  let dpi = o.dpi;
  if (mmToPx(sideMm, dpi) > MAX_SIDE) dpi = Math.floor((MAX_SIDE / sideMm) * 25.4);
  return { sideMm, dpi, sidePx: Math.round(mmToPx(sideMm, dpi)) };
}

export async function renderPrintSheet(
  image: HTMLImageElement,
  photoWidth: number,
  crop: Crop,
  o: PrintOptions,
): Promise<Blob> {
  const { dpi, sidePx } = sheetInfo(o);
  const vs = image.naturalWidth / photoWidth;
  const [cx, cy, cs] = [crop[0] * vs, crop[1] * vs, crop[2] * vs];
  const cv = document.createElement("canvas");
  cv.width = cv.height = sidePx;
  const ctx = cv.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, sidePx, sidePx);

  if (o.layout === "seamless") {
    ctx.drawImage(image, cx, cy, cs, cs, 0, 0, sidePx, sidePx);
  } else {
    const tile = mmToPx(o.tileMm, dpi);
    const gap = mmToPx(o.gapMm, dpi);
    const bleed = Math.min(mmToPx(o.bleedMm, dpi), gap / 2);
    const src = cs / o.tiles; // source pixels per tile
    const srcBleed = (bleed / tile) * src;
    for (let r = 0; r < o.tiles; r++) {
      for (let c = 0; c < o.tiles; c++) {
        const x = gap + c * (tile + gap);
        const y = gap + r * (tile + gap);
        // Clamp the bleed at the photo's outer edge so the sheet never samples outside the crop.
        const l = c === 0 ? 0 : srcBleed, t = r === 0 ? 0 : srcBleed;
        const rr = c === o.tiles - 1 ? 0 : srcBleed, b = r === o.tiles - 1 ? 0 : srcBleed;
        const kx = tile / src;
        ctx.drawImage(
          image,
          cx + c * src - l, cy + r * src - t, src + l + rr, src + t + b,
          x - l * kx, y - t * kx, tile + (l + rr) * kx, tile + (t + b) * kx,
        );
      }
    }
  }
  const png = await new Promise<Blob>((ok, fail) => cv.toBlob((b) => (b ? ok(b) : fail(new Error("print sheet too large"))), "image/png"));
  return withDpi(png, dpi);
}

// ---- PNG physical size (pHYs chunk), so print software uses the right size ----

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function withDpi(png: Blob, dpi: number): Promise<Blob> {
  const buf = new Uint8Array(await png.arrayBuffer());
  const ppm = Math.round(dpi / 0.0254);
  const chunk = new Uint8Array(21);
  const v = new DataView(chunk.buffer);
  v.setUint32(0, 9);
  chunk.set([0x70, 0x48, 0x59, 0x73], 4); // "pHYs"
  v.setUint32(8, ppm);
  v.setUint32(12, ppm);
  chunk[16] = 1; // unit: metre
  v.setUint32(17, crc32(chunk.subarray(4, 17)));
  const ihdrEnd = 8 + 25; // signature + IHDR chunk
  return new Blob([buf.subarray(0, ihdrEnd), chunk, buf.subarray(ihdrEnd)], { type: "image/png" });
}
