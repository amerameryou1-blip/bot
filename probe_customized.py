#!/usr/bin/env python3
"""Colors -> Customized: find RGB inputs precisely."""
import sys, time, os, io
import numpy as np
from PIL import Image
import pytesseract
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/customized.log"

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

def snap_ocr(page, shot, tag):
    page.screenshot(path=shot)
    txt = pytesseract.image_to_string(Image.open(shot), config="--psm 6").strip().replace("\n", " | ")
    log(f"--- {tag} OCR: {txt[:300]}")
    return txt

def dump_inputs(page, tag):
    els = page.evaluate("""() => Array.from(document.querySelectorAll('input')).map((el, i) => {
        const r = el.getBoundingClientRect();
        return {i, v: el.value, t: el.type, ph: el.placeholder||'', x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)};
    })""")
    log(f"--- {tag}: {len(els)} inputs ---")
    for e in els[:20]:
        log(f"  #{e['i']} <{e['t']}> v={e['v']!r} ph={e['ph']!r} at ({e['x']},{e['y']})")
    return els

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=FLAGS)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    page = ctx.new_page()
    page.goto("https://territorial.io/", timeout=60000, wait_until="domcontentloaded")
    time.sleep(7)
    page.bring_to_front()
    try:
        page.fill('input[placeholder*="Kingdom"], input', "CBot")
    except Exception:
        pass
    time.sleep(1)
    page.mouse.click(714, 411)  # Custom Scenario
    time.sleep(3.5)
    page.mouse.click(1115, 242)  # Colors Settings
    time.sleep(3)
    snap_ocr(page, f"{OUT}/c1.png", "after Colors Settings")

    # click 'Customized' — try a few y positions below the button
    for dy in [50, 80, 110, 140, 170]:
        yy = 242 + dy
        page.mouse.click(1115, yy)
        time.sleep(2.5)
        snap_ocr(page, f"{OUT}/c_click_{yy}.png", f"click y={yy}")
        inputs = dump_inputs(page, f"inputs y={yy}")
        # if a color picker opened (more inputs), break
        if len(inputs) > 1 or any('color' in i['t'] for i in inputs):
            log(">>> found color inputs!")
            break
        page.keyboard.press("Escape")
        time.sleep(1.5)
        # reopen colors if we closed it
        page.mouse.click(1115, 242)
        time.sleep(2.5)
    browser.close()
