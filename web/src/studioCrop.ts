// Studio engine framing: crop on the faces, tighter on small boards (the face needs more studs
// to stay readable there). Same rule as engine/studio.py crop_for.
import type { Crop, Photo } from "./api";

export const PLATE = 24; // studs per baseplate side

/** Faces at least 30 % of the largest face's area. */
export function keepFaces(faces: number[][]): number[][] {
  if (!faces.length) return [];
  const big = Math.max(...faces.map((f) => f[2] * f[3]));
  return faces.filter((f) => f[2] * f[3] >= big * 0.3);
}

export function studioCrop(photo: Photo, cols: number, rows: number, autoFrame: boolean): Crop {
  const iw = photo.width;
  const ih = photo.height;
  const gw = cols / PLATE;
  const gh = rows / PLATE;
  const a = gw / gh;
  const h0 = Math.min(ih, iw / a);
  let ch = h0;
  let cx = iw / 2;
  let cy = ih / 2;
  const faces = keepFaces(photo.faces);
  if (faces.length && autoFrame) {
    const x0 = Math.min(...faces.map((f) => f[0]));
    const y0 = Math.min(...faces.map((f) => f[1]));
    const x1 = Math.max(...faces.map((f) => f[0] + f[2]));
    const y1 = Math.max(...faces.map((f) => f[1] + f[3]));
    const m = Math.min(gw, gh) * PLATE;
    const fr = Math.min(0.78, Math.max(0.34, 0.78 - (m - 48) * 0.0037));
    ch = Math.max((y1 - y0) / fr, ((x1 - x0) * (faces.length > 1 ? 1.5 : 1.12)) / a);
    cx = (x0 + x1) / 2;
    cy = (y0 + y1) / 2 + ch * 0.08;
  }
  ch = Math.min(h0, ch);
  const cw = ch * a;
  return [Math.min(Math.max(cx - cw / 2, 0), iw - cw), Math.min(Math.max(cy - ch / 2, 0), ih - ch), cw, ch];
}
