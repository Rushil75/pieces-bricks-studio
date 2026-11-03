"""Generate on our page (All sizes) in headless Chrome and compare every board with a reference
capture (tools/ref_capture.py) brick by brick.  python tools/ours_capture_all.py OUR_URL PHOTO REF.json"""
import json
import sys

from playwright.sync_api import sync_playwright

url, photo, ref_path = sys.argv[1:4]
ref = json.load(open(ref_path))
codes = [c["code"] for c in json.load(open("config/palette_40.json"))["colors"]]
got, logs = {}, []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome")
    pg = b.new_page(viewport={"width": 1600, "height": 1100})
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.on("response", lambda r: got.__setitem__((lambda m: (len(m["grid"][0]), len(m["grid"])))(r.json()), r.json())
          if "/api/convert" in r.url and r.ok else None)
    pg.goto(url)
    pg.set_input_files("input[type=file]", photo)
    pg.wait_for_selector(".crop-canvas")
    pg.wait_for_function("!document.querySelector('.busy')", timeout=120000)
    pg.get_by_role("button", name="Generate mosaic").click()
    pg.wait_for_function("document.querySelectorAll('.size-result canvas').length === document.querySelectorAll('.size-result').length"
                         " && document.querySelectorAll('.size-result').length > 0", timeout=300000)
    pg.wait_for_timeout(500)
    pg.locator(".size-result").first.click()
    pg.wait_for_selector(".big-view canvas")
    pg.screenshot(path="output/studio_big.png")
    b.close()
for board in ref["boards"]:
    w, h = board["plates"]
    m = got.get((w * 24, h * 24))
    if not m:
        print(f"{w}x{h}: not generated")
        continue
    ours = [codes.index(c) for row in m["grid"] for c in row]
    same = sum(a == c for a, c in zip(ours, board["idx"])) / len(ours) * 100
    print(f"{w}x{h} plates ({w * 24}x{h * 24}): same bricks {same:6.2f}%")
print("\n".join(logs) or "no page errors")
