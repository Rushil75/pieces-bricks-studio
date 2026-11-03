"""Run the reference demo page in headless Chrome on a photo and save its brick grids (JSON).

  python tools/ref_capture.py URL PHOTO OUT.json [json-settings]

settings: element id -> value, e.g. '{"sat": -2}' (ids: max con sat smo lc lin sharp dith bg).
Saves, per board size: plates (w, h), crop box, faces, palette indices of every stud.
"""
import json
import sys

from playwright.sync_api import sync_playwright

url, photo, out = sys.argv[1:4]
settings = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome")
    pg = b.new_page(viewport={"width": 1400, "height": 1000})
    pg.goto(url)
    pg.set_input_files("#file", photo)
    pg.wait_for_function("/found|unavailable/i.test(document.getElementById('msg').textContent)", timeout=120000)
    for k, v in settings.items():
        pg.evaluate(f"(() => {{ const e = document.getElementById('{k}'); e.value = {json.dumps(v)}; e.oninput(); }})()")
    pg.wait_for_timeout(1500)
    data = pg.evaluate("""() => ({
        msg: document.getElementById('msg').textContent,
        faces: faces,
        image: [img.width, img.height],
        boards: sizes.map(([w, h], i) => ({plates: [w, h], crop: cropFor(w, h), idx: Array.from(res[i])})),
    })""")
    b.close()
json.dump(data, open(out, "w"))
print(data["msg"], "| faces:", len(data["faces"]), "| boards:", [b["plates"] for b in data["boards"]])
