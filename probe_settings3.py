#!/usr/bin/env python3
"""Accept data-usage popup -> reopen Settings -> find color option."""
import sys, time, os, io
import numpy as np
from PIL import Image
import pytesseract
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/settings3.log"

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
        document.querySelectorAll('button, input, canvas, select, [role=button], [onclick]').forEach((el, i) => {
            const r = el.getBoundingClientRect();
            if (r.width > 0 && r.height > 0)
                out.push({i, tag: el.tagName, cls: (el.className||'').toString().slice(0,30),
                          text: (el.innerText || el.value || el.placeholder || '').trim().slice(0, 40),
                          x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2), w: Math.round(r.width), h: Math.round(r.height)});
        });
        return out.slice(0, 120);
    }""")
    log(f"--- {tag}: {len(els)} elements ---")
    for e in els[:100]:
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
        page.fill('input[placeholder*="Kingdom"], input', "SetBot")
    except Exception:
        pass
    time.sleep(1)

    # Game Menu -> Settings (this may show the data-usage popup)
    page.mouse.click(714, 499); time.sleep(2.5)
    els = dump_els(page, "game menu")
    set_btn = next((e for e in els if 'Settings' in e['text']), None)
    page.mouse.click(set_btn['x'], set_btn['y']); time.sleep(2.5)
    els = dump_els(page, "after settings click")
    acc = next((e for e in els if 'Accept' in e['text']), None)
    if acc:
        page.mouse.click(acc['x'], acc['y']); log("clicked Accept"); time.sleep(2)
        # reopen settings
        page.mouse.click(set_btn['x'], set_btn['y']); time.sleep(3)
        els = dump_els(page, "settings after accept")

    page.screenshot(path=f"{OUT}/settings3.png")
    txt = pytesseract.image_to_string(Image.open(f"{OUT}/settings3.png"), config="--psm 6").strip()
    log("OCR settings:\n" + txt[:1200])
    browser.close()
