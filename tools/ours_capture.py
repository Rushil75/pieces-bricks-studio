"""Generate every square Studio board on our page in headless Chrome and compare with a reference
capture (tools/ref_capture.py) brick by brick.

  python tools/ours_capture.py OUR_URL PHOTO REF.json [--auto-frame]

Without --auto-frame the face steps are switched off, for photos where the reference found no face.
"""
import json
import sys

from playwright.sync_api import sync_playwright

url, photo, ref_path = sys.argv[1:4]
auto_frame = "--auto-frame" in sys.argv
ref = json.load(open(ref_path))
codes = [c["code"] for c in json.load(open("config/palette_40.json"))["colors"]]
logs = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome")
    pg = b.new_page(viewport={"width": 1600, "height": 1100})
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.goto(url)
    pg.wait_for_selector(".drop")
    pg.set_input_files("input[type=file]", photo)
    pg.wait_for_selector(".crop-canvas", timeout=60000)
    pg.wait_for_function("!document.querySelector('.busy')", timeout=120000)
    if not auto_frame:
        # Reference found no face: no auto-frame, no face brightening, no background simplify
        pg.locator(".checks label", has_text="Auto-frame").locator("input").uncheck()
        pg.locator(".checks label", has_text="Brighten face").locator("input").uncheck()
        pg.locator(".slider.adj", has_text="Background simplify").locator("input").fill("0")
    first = True
    for board in ref["boards"]:
        w, h = board["plates"]
        pg.locator(".size-card", has_text=f"{w}×{h} plates").first.click()
        with pg.expect_response(lambda r: "/api/convert" in r.url, timeout=180000) as resp:
            if first:
                pg.get_by_role("button", name="Generate mosaic").click()
                first = False
        m = resp.value.json()
        ours = [codes.index(c) for row in m["grid"] for c in row]
        same = sum(a == b for a, b in zip(ours, board["idx"])) / len(ours) * 100
        print(f"{w}x{h} plates ({w * 24}x{h * 24}): crop {[round(v) for v in m['settings']['crop_resolved']]} "
              f"ref {[round(board['crop'][k]) for k in 'xywh']}  same bricks {same:6.2f}%")
        pg.wait_for_timeout(300)
    pg.locator(".panel.left section").nth(2).screenshot(path="output/studio_settings.png")
    pg.locator(".stage").screenshot(path="output/studio_mosaic.png")
    b.close()
print("\n".join(logs) or "no page errors")
