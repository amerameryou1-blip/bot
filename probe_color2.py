#!/usr/bin/env python3
"""Click the color-grid icon (bottom-left canvas) and find how to set RGB."""
import sys, time, os, io
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/color2.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

FLAGS = [
    "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=swiftshader", "--disable-gpu",
    "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
    "--disable-background-timer-throttling", "--disable-features=CalculateNativeWinOcclusion",
    "--enable-unsafe-swiftshader",
]

def dump_els(page, tag):
    els = page.evaluate("""() => {
        const out = [];
        document.querySelectorAll('button, input, canvas, [role=button], [onclick]').forEach((el, i) => {
            const r = el.getBoundingClientRect();
            if (r.width > 0 && r.height > 0)
                out.push({i, tag: el.tagName, cls: (el.className||'').toString().slice(0,30),
                          text: (el.innerText || el.value || el.placeholder || '').trim().slice(0, 40),
                          x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2), w: Math.round(r.width), h: Math.round(r.height)});
        });
        return out.slice(0, 100);
    }""")
    log(f"--- {tag}: {len(els)} elements ---")
    for e in els[:80]:
        log(f"  #{e['i']} <{e['tag']}> cls={e['cls']!r} '{e['text']}' at ({e['x']},{e['y']}) {e['w']}x{e['h']}")
    return els

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=FLAGS)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    page = ctx.new_page()
    page.goto("https://territorial.io/", timeout=60000, wait_until="domcontentloaded")
    time.sleep(7)
    page.bring_to_front()
    try:
        page.fill('input[placeholder*="Kingdom"], input', "ColorBot")
    except Exception:
        pass

    # click the color-grid canvas (bottom-left)
    page.mouse.click(57, 742)
    time.sleep(3)
    page.screenshot(path=f"{OUT}/color_picker.png")
    dump_els(page, "after color icon click")

    # try clicking again if nothing changed
    page.screenshot(path=f"{OUT}/color_picker2.png")
    browser.close()
