"""Automated quality checks from the brief: palette-only, exact counts, determinism,
and outputs regenerated from matrix.json only."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import pipeline  # noqa: E402
from engine.palette import load_client_palette  # noqa: E402

SIZE = 32  # small grid keeps the suite fast; the logic is size-independent


@pytest.fixture(scope="module", params=[("p5", "adaptive", False), ("p5", "client", False), ("p4", "adaptive", True)])
def result(request, tmp_path_factory):
    job, palette, dither = request.param
    photo = pipeline.ROOT / "input" / pipeline.job_settings(job)["file"]
    if not photo.exists():
        pytest.skip(f"client photo {photo.name} not present (photos are kept out of the repo)")
    out = tmp_path_factory.mktemp(f"{job}_{palette}_{dither}")
    return pipeline.run(job, SIZE, dither, {"palette": palette}, out)


def test_every_cell_is_a_palette_code(result):
    m = result["matrix"]
    allowed = {e["code"] for e in m["palette"]["colors"]}
    cells = {c for row in m["grid"] for c in row}
    assert cells <= allowed
    if m["palette"]["name"] == "client":
        assert cells <= set(load_client_palette().codes)
        assert all(re.fullmatch(r"C(0[1-9]|[1-3]\d|40)", c) for c in cells)
    assert all(re.fullmatch(r"#[0-9A-F]{6}", e["hex"]) for e in m["palette"]["colors"])


def test_counts_sum_to_n_squared_and_match_matrix(result):
    d = result["dir"]
    matrix = json.loads((d / "matrix.json").read_text())
    assert len(matrix["grid"]) == SIZE and all(len(r) == SIZE for r in matrix["grid"])
    with open(d / "counts.csv") as f:
        rows = list(csv.DictReader(f))
    total = rows.pop()
    assert total["Code"] == "TOTAL" and int(total["Quantity"]) == SIZE * SIZE
    counts = {r["Code"]: int(r["Quantity"]) for r in rows}
    assert sum(counts.values()) == SIZE * SIZE
    assert counts == dict(Counter(c for row in matrix["grid"] for c in row))


def test_same_settings_give_identical_matrix(result, tmp_path):
    m = result["matrix"]
    pipeline._preprocess_cached.cache_clear()
    again = pipeline.run(m["job"], SIZE, m["settings"]["dither"], {"palette": m["palette"]["name"]}, tmp_path)
    assert (result["dir"] / "matrix.json").read_bytes() == (again["dir"] / "matrix.json").read_bytes()


def test_outputs_are_generated_from_matrix_json(result, tmp_path):
    """Editing matrix.json and re-rendering must change preview, grid and counts."""
    d = result["dir"]
    m = json.loads((d / "matrix.json").read_text())
    new_code = m["palette"]["colors"][0]["code"]
    m["grid"] = [[new_code] * SIZE for _ in range(SIZE)]
    (tmp_path / "matrix.json").write_text(json.dumps(m))
    pipeline.write_outputs(tmp_path)
    with open(tmp_path / "counts.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["Code"] == new_code and int(rows[0]["Quantity"]) == SIZE * SIZE
    for name in ("preview.png", "flat.png", "grid.png"):
        assert (tmp_path / name).read_bytes() != (d / name).read_bytes()
