export type Crop = [number, number, number]; // x, y, size in original pixels

export interface Photo {
  id: string;
  width: number;
  height: number;
  view_url: string;
  faces: [number, number, number, number][];
  auto_crop: Crop;
  title?: string;
}

export interface PaletteColor {
  code: string;
  name: string;
  hex: string;
}

export interface Matrix {
  size: number;
  palette_version: string;
  palette: { name: string; colors: PaletteColor[] };
  palette_full: PaletteColor[];
  settings: Record<string, unknown>;
  variant: string;
  edited?: boolean;
  similarity?: Similarity;
  grid: string[][];
}

export interface Similarity {
  similarity: number;
  mean_delta_e: number;
  pct_unnoticeable: number;
  pct_close: number;
  structure_ssim: number;
  face_mean_delta_e?: number;
}

export interface ConvertOptions {
  size: number;
  dither: boolean;
  palette: "adaptive" | "client";
  colors: number;
  face_detail: number;
  background: "bilateral" | "meanshift" | "none";
}

// Where the Python API lives. Empty = same origin (local server). On Netlify set VITE_API_BASE
// to the Render URL, e.g. https://pieces-bricks-api.onrender.com
const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const PW_KEY = "pb-demo-password";

export const apiUrl = (path: string) => BASE + path;

export class PasswordRequired extends Error {}

function password(): string {
  try {
    return sessionStorage.getItem(PW_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setPassword(pw: string) {
  try {
    sessionStorage.setItem(PW_KEY, pw);
  } catch {
    /* private mode: password lasts for this page only */
  }
  memoryPw = pw;
}
let memoryPw = "";

function call(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const pw = password() || memoryPw;
  if (pw) headers.set("X-Demo-Password", pw);
  return fetch(apiUrl(path), { ...init, headers });
}

export async function health(): Promise<{ ok: boolean; password: boolean }> {
  return json(await fetch(apiUrl("/api/health")));
}

async function json<T>(res: Response): Promise<T> {
  if (res.status === 401) throw new PasswordRequired("Password required");
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function uploadPhoto(file: File): Promise<Photo> {
  const fd = new FormData();
  fd.append("file", file);
  return json(await call("/api/upload", { method: "POST", body: fd }));
}

export async function listSamples(): Promise<{ key: string; title: string }[]> {
  return json(await call("/api/samples"));
}

export async function loadSample(key: string): Promise<Photo> {
  return json(await call(`/api/samples/${key}`, { method: "POST" }));
}

export async function clientPalette(): Promise<{ version: string; colors: PaletteColor[] }> {
  return json(await call("/api/palette"));
}

// The photo goes with every conversion: serverless hosts don't keep uploads between requests.
export async function convert(photo: Blob, crop: Crop, o: ConvertOptions, job?: string): Promise<Matrix> {
  const fd = new FormData();
  fd.append("file", photo, "photo.jpg");
  fd.append("options", JSON.stringify({ crop, job: job ?? null, ...o }));
  return json(await call("/api/convert", { method: "POST", body: fd }));
}

export async function sampleFile(key: string): Promise<Blob> {
  const res = await call(`/api/samples/${key}/file`);
  if (!res.ok) await json(res);
  return res.blob();
}

const MAX_SIDE = 3000; // far more than the engine needs per brick
const MAX_BYTES = 4_000_000; // hosted requests are capped at 4.5 MB

/** EXIF-rotate and shrink a photo so it uploads fast and fits the request limit. */
export async function preparePhoto(file: Blob): Promise<Blob> {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  let side = Math.min(MAX_SIDE, Math.max(bmp.width, bmp.height));
  if (file.size <= MAX_BYTES && side === Math.max(bmp.width, bmp.height) && file.type === "image/jpeg") {
    bmp.close();
    return file;
  }
  for (let quality = 0.95; ; ) {
    const k = side / Math.max(bmp.width, bmp.height);
    const cv = document.createElement("canvas");
    cv.width = Math.round(bmp.width * k);
    cv.height = Math.round(bmp.height * k);
    const ctx = cv.getContext("2d")!;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bmp, 0, 0, cv.width, cv.height);
    const blob = await new Promise<Blob>((ok) => cv.toBlob((b) => ok(b!), "image/jpeg", quality));
    if (blob.size <= MAX_BYTES || side < 800) {
      bmp.close();
      return blob;
    }
    if (quality > 0.86) quality -= 0.04;
    else side = Math.round(side * 0.85);
  }
}

export async function exportPack(matrix: Matrix): Promise<Blob> {
  const res = await call("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ matrix }),
  });
  if (!res.ok) await json(res);
  return res.blob();
}
