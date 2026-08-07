#!/usr/bin/env python3
"""Try interacting with the RGB fields (click + type + Enter)."""
import sys, time, os, io
import numpy as np
from PIL import Image
import pytesseract
from playwright.sync_api import sync_playwright

OUT = "/home/user/runs"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/rgb_input.log"

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

def ocr(page, tag):
    shot = f"{OUT}/r_{tag}.png"
    page.screenshot(path=shot)
    txt = pytesseract.image_to_string(Image.open(shot), config="--psm 11").strip().replace("\n", " | ")[:200]
    log(f"--- {tag}: {txt}")
    return txt

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=FLAGS)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    page = ctx.new_page()
    page.goto("https://territorial.io/", timeout=60000, wait_until="domcontentloaded")
    time.sleep(7)
    page.bring_to_front()
    try:
        page.fill('input[placeholder*="Kingdom"], input', "RGBBot")
    except Exception:
        pass
    time.sleep(1)
    page.mouse.click(714, 411)  # Custom Scenario
    time.sleep(3.5)
    page.mouse.click(1115, 242)  # Colors Settings
    time.sleep(3)
    page.mouse.click(50, 204)    # Customized
    time.sleep(3)
    ocr(page, "customized")

    # click the first field (351,187), select-all, type 255
    page.mouse.click(351, 187)
    time.sleep(1)
    page.keyboard.press("Control+A")
    page.keyboard.type("255")
    time.sleep(0.5)
    page.keyboard.press("Enter")
    time.sleep(1.5)
    ocr(page, "after typing 255 in field 1")

    # click the second field (351,211) type 60
    page.mouse.click(351, 211)
    time.sleep(0.5)
    page.keyboard.press("Control+A")
    page.keyboard.type("60")
    page.keyboard.press("Enter")
    time.sleep(1.5)
    ocr(page, "after typing 60 in field 2")

    # click third field (351,235) type 60
    page.mouse.click(351, 235)
    time.sleep(0.5)
    page.keyboard.press("Control+A")
    page.keyboard.type("60")
    page.keyboard.press("Enter")
    time.sleep(1.5)
    ocr(page, "after typing 60 in field 3")
    page.screenshot(path=f"{OUT}/rgb_set.png")
    browser.close()
