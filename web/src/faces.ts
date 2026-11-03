// Studio engine face detection in the browser, with the same detector and settings as the
// reference demo: face-api's TinyFaceDetector (input 512, score >= 0.45), a second pass over
// four overlapping 60 % tiles when nothing is found, overlap removal, and faces at least 30 % of
// the largest one. Loaded from the CDN on first use; callers fall back to the server's faces.
const LIB = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/";

type Face = { x: number; y: number; w: number; h: number; s: number };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FaceApi = any;

let loading: Promise<FaceApi> | null = null;

function faceApi(): Promise<FaceApi> {
  if (!loading) {
    loading = new Promise<FaceApi>((ok, fail) => {
      const el = document.createElement("script");
      el.src = LIB + "dist/face-api.js";
      el.onload = async () => {
        try {
          const fa = (window as unknown as { faceapi: FaceApi }).faceapi;
          await fa.tf.ready();
          await fa.nets.tinyFaceDetector.loadFromUri(LIB + "model/");
          ok(fa);
        } catch (e) {
          fail(e);
        }
      };
      el.onerror = () => fail(new Error("could not download the face detector"));
      document.head.append(el);
    }).catch((e) => {
      loading = null;
      throw e;
    });
  }
  return loading;
}

async function scan(fa: FaceApi, img: HTMLImageElement, sx: number, sy: number, sw: number, sh: number): Promise<Face[]> {
  const k = Math.min(1, 1024 / Math.max(sw, sh));
  const c = document.createElement("canvas");
  c.width = Math.round(sw * k);
  c.height = Math.round(sh * k);
  c.getContext("2d")!.drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
  const found = await fa.detectAllFaces(c, new fa.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.45 }));
  return found.map((d: { box: { x: number; y: number; width: number; height: number }; score: number }) => ({
    x: sx + d.box.x / k,
    y: sy + d.box.y / k,
    w: d.box.width / k,
    h: d.box.height / k,
    s: d.score,
  }));
}

function iou(a: Face, b: Face): number {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  if (w <= 0 || h <= 0) return 0;
  const i = w * h;
  return i / (a.w * a.h + b.w * b.h - i);
}

/** Face boxes [x, y, w, h] in the image's natural pixels. Throws if the detector can't load. */
export async function detectFaces(img: HTMLImageElement): Promise<number[][]> {
  const fa = await faceApi();
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  let found = await scan(fa, img, 0, 0, W, H);
  if (!found.length) {
    for (const oy of [0, 0.4]) for (const ox of [0, 0.4]) found.push(...(await scan(fa, img, W * ox, H * oy, W * 0.6, H * 0.6)));
  }
  found.sort((a, b) => b.s - a.s);
  const unique: Face[] = [];
  for (const f of found) if (!unique.some((g) => iou(f, g) > 0.3)) unique.push(f);
  const big = Math.max(0, ...unique.map((f) => f.w * f.h));
  return unique.filter((f) => f.w * f.h >= big * 0.3).map((f) => [f.x, f.y, f.w, f.h]);
}
