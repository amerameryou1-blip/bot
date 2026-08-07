#!/usr/bin/env python3
"""Game Menu -> Settings -> find color options."""
import sys, time, os, io
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/settings_probe.log"

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
    page.mouse.click(714, 499)  # Game Menu
    time.sleep(2.5)
    # click Settings at (534,295)
    page.mouse.click(534, 295)
    time.sleep(2.5)
    dump_els(page, "Settings")
    page.screenshot(path=f"{OUT}/settings.png")
    browser.close()
