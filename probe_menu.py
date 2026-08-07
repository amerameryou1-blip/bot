#!/usr/bin/env python3
"""Explore Game Menu + the bottom-left icon for color settings."""
import sys, time, os, io
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/menu_probe.log"

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

def click(page, x, y, label):
    page.mouse.click(x, y)
    log(f"clicked {label} ({x},{y})")
    time.sleep(2.5)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=FLAGS)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    page = ctx.new_page()
    page.goto("https://territorial.io/", timeout=60000, wait_until="domcontentloaded")
    time.sleep(7)
    page.bring_to_front()
    try:
        page.fill('input[placeholder*="Kingdom"], input', "MenuBot")
    except Exception:
        pass
    time.sleep(1)

    # Try the Game Menu (☰) at (714,499)
    click(page, 714, 499, "Game Menu")
    dump_els(page, "after Game Menu")
    page.screenshot(path=f"{OUT}/menu1.png")
    # Escape to close
    page.keyboard.press("Escape"); time.sleep(1)

    # Try clicking the bottom-left icon at several points (maybe it's a dropdown)
    for pt in [(57, 742), (99, 777), (20, 770), (57, 770)]:
        before = np.array(Image.open(io.BytesIO(page.screenshot())).convert('RGB'))
        page.mouse.click(pt[0], pt[1])
        time.sleep(2)
        after = np.array(Image.open(io.BytesIO(page.screenshot())).convert('RGB'))
        d = float(np.abs(before.astype(int) - after.astype(int)).mean())
        log(f"icon click ({pt[0]},{pt[1]}): diff={d:.2f}")
        if d > 1.0:
            dump_els(page, f"after icon click ({pt[0]},{pt[1]})")
            page.screenshot(path=f"{OUT}/icon_{pt[0]}_{pt[1]}.png")
            page.keyboard.press("Escape"); time.sleep(1)
    browser.close()
