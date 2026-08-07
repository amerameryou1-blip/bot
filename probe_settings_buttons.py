#!/usr/bin/env python3
"""Click each editor Settings button, OCR the panel, find color controls."""
import sys, time, os, io
import numpy as np
from PIL import Image
import pytesseract
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/settings_buttons.log"

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

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=FLAGS)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    page = ctx.new_page()
    page.goto("https://territorial.io/", timeout=60000, wait_until="domcontentloaded")
    time.sleep(7)
    page.bring_to_front()
    try:
        page.fill('input[placeholder*="Kingdom"], input', "SBot")
    except Exception:
        pass
    time.sleep(1)
    page.mouse.click(714, 411)  # Custom Scenario
    time.sleep(3)

    # collect Settings buttons
    els = page.evaluate("""() => Array.from(document.querySelectorAll('button')).map((el, i) => {
        const r = el.getBoundingClientRect();
        return {i, text: (el.innerText||'').trim().slice(0,30), x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)};
    })""")
    settings = [e for e in els if 'Settings' in e['text'] or 'More' in e['text']]
    log(f"found {len(settings)} settings-ish buttons")
    for idx, s in enumerate(settings):
        page.mouse.click(s['x'], s['y'])
        time.sleep(2.5)
        shot = f"{OUT}/sbtn_{idx}.png"
        page.screenshot(path=shot)
        txt = pytesseract.image_to_string(Image.open(shot), config="--psm 6").strip().replace("\n", " | ")[:300]
        log(f"btn#{idx} '{s['text']}' at ({s['x']},{s['y']}): {txt}")
        # check for DOM inputs (RGB fields?)
        inputs = page.evaluate("""() => Array.from(document.querySelectorAll('input')).map(i => ({v: i.value, ph: i.placeholder||'', t: i.type})).filter(x => x.ph || (x.v && isNaN(Number(x.v))))""")
        if inputs:
            log(f"   inputs: {inputs[:8]}")
        page.keyboard.press("Escape")
        time.sleep(1)
    browser.close()
