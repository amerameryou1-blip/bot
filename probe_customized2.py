#!/usr/bin/env python3
"""Click 'Customized' at (50,204), then find the color inputs."""
import sys, time, os, io
import numpy as np
from PIL import Image, ImageOps
import pytesseract
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/customized2.log"

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

def ocr_words(page):
    shot = f"{OUT}/tmp.png"
    page.screenshot(path=shot)
    img = Image.open(shot)
    img3 = img.resize((img.width*2, img.height*2), Image.LANCZOS)
    data = pytesseract.image_to_data(ImageOps.grayscale(img3), config='--psm 11', output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(data['text'])):
        t = data['text'][i].strip()
        if t and len(t) > 1:
            out.append((t, data['left'][i]//2, data['top'][i]//2))
    return out

def dump_inputs(page, tag):
    els = page.evaluate("""() => Array.from(document.querySelectorAll('input')).map((el, i) => {
        const r = el.getBoundingClientRect();
        return {i, v: el.value, t: el.type, ph: el.placeholder||'', x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)};
    })""")
    log(f"--- {tag}: {len(els)} inputs ---")
    for e in els[:25]:
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
    log("words now: " + str(ocr_words(page)))

    # click 'Customized'
    page.mouse.click(50, 204)
    time.sleep(3)
    log("after Customized click, words: " + str(ocr_words(page)))
    dump_inputs(page, "inputs after Customized")
    page.screenshot(path=f"{OUT}/customized_open.png")
    browser.close()
