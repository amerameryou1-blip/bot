#!/usr/bin/env python3
"""Live play with the v2 brain (teacher or student) on territorial.io.

Bridge sim->real: zoom OUT to the full-map view (same geometry as the sim's
full-world frames), detect the map rect (corner-bg trick), map the 16x16
click grid onto it, double-click the brain's choices.

Env: BRAIN_PT=path to teacher.pt/student.pt, ARCH=teacher|student
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from nn.model_v2 import TeacherV2, StudentV2

FLAGS = ["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=swiftshader",
         "--disable-gpu"]
GRID = 16


def grab_rgb(page):
    return np.asarray(Image.open(
        __import__("io").BytesIO(page.screenshot())).convert("RGB"))


def map_rect_live(rgb):
    """Corner-bg trick from rebuild_maps: map = non-background bbox."""
    cs = np.concatenate([rgb[:5, :5].reshape(-1, 3), rgb[:5, -5:].reshape(-1, 3),
                         rgb[-5:, :5].reshape(-1, 3), rgb[-5:, -5:].reshape(-1, 3)])
    corner = np.median(cs, axis=0)
    bg = (np.abs(rgb.astype(int) - corner).max(axis=2) < 18)
    ys, xs = np.where(~bg)
    if len(ys) < 1000:
        return None
    # trim UI panels: keep the biggest dense box via percentiles
    y0, y1 = int(np.percentile(ys, 0.5)), int(np.percentile(ys, 99.5))
    x0, x1 = int(np.percentile(xs, 0.5)), int(np.percentile(xs, 99.5))
    return y0, y1, x0, x1


def preprocess(rgb, rect, prev):
    y0, y1, x0, x1 = rect
    crop = rgb[y0:y1 + 1, x0:x1 + 1]
    small = np.asarray(Image.fromarray(crop).resize((128, 128), Image.BILINEAR))
    r = small.transpose(2, 0, 1).astype(np.float32) / 255.0
    if prev is None:
        diff = np.full_like(r, 0.5)
    else:
        diff = np.clip(r - prev, -1, 1) * 0.5 + 0.5
    x = np.concatenate([r, diff], 0)[None]
    return x, r


def main():
    brain_pt = os.environ.get("BRAIN_PT", str(REPO / "weights/nn/v2/teacher.pt"))
    arch = os.environ.get("ARCH", "teacher")
    minutes = float(os.environ.get("PLAY_MINUTES", "5"))
    import torch
    net = TeacherV2() if arch == "teacher" else StudentV2()
    net.load_state_dict(torch.load(brain_pt, map_location="cpu"))
    net.eval()
    print(f"brain: {arch} from {brain_pt}", flush=True)

    with sync_playwright() as p:
        page = p.chromium.launch(headless=True, args=FLAGS).new_context(
            viewport={"width": 1280, "height": 800}).new_page()
        page.goto("https://territorial.io/", timeout=60000,
                  wait_until="domcontentloaded")
        time.sleep(7)
        try:
            page.fill('input[placeholder*="Kingdom"], input', "AureliaBot")
        except Exception:
            pass
        from bot.ui import enter_custom_match
        if not enter_custom_match(page, log=print):
            print("FATAL: no match"); return
        # spawn: double-click a land spot (center-ish of map rect)
        rgb = grab_rgb(page)
        rect = map_rect_live(rgb)
        if rect is None:
            print("FATAL: no rect"); return
        y0, y1, x0, x1 = rect
        page.mouse.dblclick((x0 + x1) // 2, (y0 + y1) // 2)
        time.sleep(1.5)
        # zoom fully out: hit the minus button several times
        for _ in range(6):
            page.mouse.click(1240, 443)
            time.sleep(0.5)
        time.sleep(1)
        prev = None
        start = time.time()
        last_rank = None
        while time.time() - start < minutes * 60:
            rgb = grab_rgb(page)
            rect = map_rect_live(rgb)
            if rect is None:
                time.sleep(1); continue
            x, prev = preprocess(rgb, rect, prev)
            ctx = np.zeros((1, 8), dtype=np.float32)
            with torch.no_grad():
                click, kind, pct, value = net(
                    torch.tensor(x), torch.tensor(ctx))
            ki = int(kind[0].argmax())
            if ki == 2:  # bank: do nothing this tick
                time.sleep(0.4); continue
            cell = int(click[0].argmax())
            cy, cx = divmod(cell, GRID)
            y0, y1, x0, x1 = rect
            sx = x0 + (cx + 0.5) / GRID * (x1 - x0)
            sy = y0 + (cy + 0.5) / GRID * (y1 - y0)
            if ki == 1:  # attack: set slider low-ish first
                for _ in range(3):
                    page.keyboard.press("s")
            page.mouse.dblclick(sx, sy)
            time.sleep(0.45)
        print("LIVE SESSION DONE", flush=True)


if __name__ == "__main__":
    main()
