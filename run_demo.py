"""CLI.

  python run_demo.py --photo p5 --size 48 --dither off
  python run_demo.py --all                       # every job, size and variant + compare sheets
  python run_demo.py --photo p5 --palette client # the client's fixed 58 colours
"""
import argparse
import time

from engine import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", help="job id from config/photos.yaml (p1, p2a, p2b, p3a, p3b, p4, p5)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--size", type=int, help="grid size N (default: sizes listed in photos.yaml)")
    ap.add_argument("--dither", choices=["on", "off", "both"], default="both")
    ap.add_argument("--palette", choices=["adaptive", "client"])
    ap.add_argument("--colors", type=int, help="adaptive palette size")
    ap.add_argument("--out", help="output root (default: output/)")
    a = ap.parse_args()

    cfg = pipeline.load_config()
    jobs = list(cfg["jobs"]) if a.all else [a.photo]
    if not jobs or jobs == [None]:
        ap.error("give --photo or --all")
    variants = {"on": [True], "off": [False], "both": [False, True]}[a.dither]
    overrides = {"palette": a.palette, "colors": a.colors}

    for job in jobs:
        for size in [a.size] if a.size else cfg["jobs"][job]["sizes"]:
            for d in variants:
                t = time.time()
                r = pipeline.run(job, size, d, overrides, a.out)
                sim = r["matrix"]["similarity"]
                print(f"{r['dir'].name:28s} colours={len(r['matrix']['palette']['colors']):3d}  "
                      f"similarity={sim['similarity']:5.1f}%  dE={sim['mean_delta_e']:.2f}  {time.time() - t:5.1f}s")
            if len(variants) == 2:
                print("  ->", pipeline.write_compare(job, size, a.out).name)


if __name__ == "__main__":
    main()
