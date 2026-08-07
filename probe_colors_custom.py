#!/usr/bin/env python3
"""Colors setting -> Customized -> find RGB inputs."""
import sys, time, os, io
import numpy as np
from PIL import Image
import pytesseract
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/colors_custom.log"

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

def dump(page, tag, shot):
    els = page.evaluate("""() => Array.from(document.querySelectorAll('button, input, canvas')).map((el, i) => {
        const r = el.getBoundingClientRect();
        return {i, tag: el.tagName, text: (el.innerText||el.value||el.placeholder||'').trim().slice(0,30),
                x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2), w: Math.round(r.width), h: Math.round(r.height)};
    })""")
    log(f"--- {tag}: {len(els)} els ---")
    for e in els[:60]:
        log(f"  #{e['i']} <{e['tag']}> '{e['text']}' at ({e['x']},{e['y']}) {e['w']}x{e['h']}")
    page.screenshot(path=shot)
    txt = pytesseract.image_to_string(Image.open(shot), config="--psm 6").strip().replace("\n", " | ")[:400]
    log(f"  OCR: {txt}")

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
    time.sleep(3)
    page.mouse.click(1115, 242)  # Colors Settings
    time.sleep(2.5)
    dump(page, "colors panel", f"{OUT}/colors_panel.png")

    # click Customized (find it by OCR-ish position: it's a radio-ish option)
    # From previous OCR: "Colors | Options | @ Random | @ Customized | @ Back"
    # Try clicking where 'Customized' is — probe positions
    for y in [242+60, 242+90, 242+120, 242+150]:
        page.mouse.click(1115, y)
        time.sleep(2)
        shot = f"{OUT}/colors_click_{y}.png"
        txt = pytesseract.image_to_string(Image.open(shot) if False else Image.fromarray(np.array(Image.open(shot))), config="--psm 6").strip().replace("\n", " | ")[:200] if False else ""
        dump(page, f"click y={y}", shot)
        # if a new panel with inputs appears, stop
        inputs = page.evaluate("""() => Array.from(document.querySelectorAll('input')).map(i => ({v:i.value, t:i.type, ph:i.placeholder||''}))""")
        log(f"   inputs now: {inputs[:10]}")
        if any(i['ph'] for i in inputs):
            log("   >>> FOUND custom inputs")
            break
        page.keyboard.press("Escape"); time.sleep(1)
    browser.close()
