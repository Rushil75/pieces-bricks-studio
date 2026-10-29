"""counts.csv and matrix.json. Everything downstream is generated from matrix.json."""
import csv
import io
import json
from collections import Counter
from pathlib import Path


def build_matrix(grid_codes, palette, settings, extra=None):
    n = len(grid_codes)
    used = set(c for row in grid_codes for c in row)
    return {
        "size": n,
        "palette_version": palette.version,
        "palette": {
            "name": palette.name,
            "colors": [e for e in palette.entries if e["code"] in used],
        },
        "settings": settings,
        **(extra or {}),
        "grid": grid_codes,
    }


def write_matrix(matrix, path):
    Path(path).write_text(json.dumps(matrix, indent=1, sort_keys=False))


def read_matrix(path):
    return json.loads(Path(path).read_text())


def count(matrix):
    return Counter(c for row in matrix["grid"] for c in row)


def counts_csv(matrix):
    counts = count(matrix)
    info = {e["code"]: e for e in matrix["palette"]["colors"]}
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    f = io.StringIO(newline="")
    w = csv.writer(f)
    w.writerow(["Code", "Name", "HEX", "Quantity"])
    for code, q in rows:
        w.writerow([code, info[code]["name"], info[code]["hex"], q])
    w.writerow(["TOTAL", "", "", sum(counts.values())])
    return f.getvalue()


def write_counts(matrix, path):
    Path(path).write_text(counts_csv(matrix), newline="")
