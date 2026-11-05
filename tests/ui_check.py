"""End-to-end interface check (AC-09) with Playwright.

Usage: python tests/ui_check.py http://localhost:8080 <api-key> out_dir
       python tests/ui_check.py file-server-url-of-demo "" out_dir   (in-browser demo)
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

url, key, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url)
    pg.wait_for_selector("#samples button", timeout=30000)
    mode = pg.inner_text("#modeBadge")
    if key:
        pg.fill("#apiKey", key)
    if "demo" in mode.lower():
        pg.wait_for_function("document.getElementById('connStatus').textContent.includes('ready')", timeout=180000)
    pg.click("#samples button:nth-child(4)")
    pg.wait_for_function("!document.getElementById('measureBtn').disabled", timeout=30000)
    # draw a parcel: 4 clicks then close on the first point
    box = pg.locator("#cv").bounding_box()
    pts = [(0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)]
    pg.click("#drawBtn")
    for fx, fy in pts:
        pg.mouse.click(box["x"] + fx * box["width"], box["y"] + fy * box["height"])
    pg.mouse.click(box["x"] + pts[0][0] * box["width"], box["y"] + pts[0][1] * box["height"])
    pg.click("#measureBtn")
    pg.wait_for_function("document.getElementById('runStatus').textContent.startsWith('Done')", timeout=300000)
    pg.screenshot(path=str(out / "measured.png"), full_page=True)
    lawn = pg.inner_text("#kLawn")
    total = pg.inner_text("#kQuote")
    acc = pg.inner_text("#accTbl")
    # override lawn -> quote changes
    pg.fill("input.override[data-k=lawn_sqft]", "20000")
    pg.dispatch_event("input.override[data-k=lawn_sqft]", "change")
    pg.wait_for_timeout(800)
    total2 = pg.inner_text("#kQuote")
    print({"mode": mode, "status": pg.inner_text("#runStatus"), "lawn_sqft": lawn, "quote": total,
           "quote_after_override": total2, "export_status": pg.inner_text("#exStatus"), "errors": errors})
    print(acc)
    b.close()
if errors or total == total2:
    sys.exit(1)
