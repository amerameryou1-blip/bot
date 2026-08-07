"""DOM helpers for the territorial.io UI — robust element finding.

The game's menu is DOM-rendered, but many items are <div>/<span>/<li>, not
<button>. A button-only scan misses "Custom Scenario" -> the bot would fall
through to the main-menu "Play" and join the MULTIPLAYER LOBBY instead of a
custom match. This module scans every element, prefers the smallest (leaf)
match, and exposes a strict Play matcher (emoji-stripped, exact).
"""
from __future__ import annotations


def _scan_elements(page, text_filter=None):
    """Return [{text,x,y,w,h}] for visible DOM elements (leaf-preferred)."""
    return page.evaluate(
        """(filter) => {
            const all = Array.from(document.querySelectorAll(
                'button, div, span, li, a, h1, h2, h3, p, label'));
            const out = [];
            const seen = new Set();
            for (const el of all) {
                const t = (el.innerText || el.textContent || '').trim();
                const r = el.getBoundingClientRect();
                if (!t || r.width < 4 || r.height < 4) continue;
                if (r.top > innerHeight || r.left > innerWidth) continue;
                // skip elements that contain an already-matched child text
                // (leaf preference): keep the smallest element per text
                const key = t + '|' + Math.round(r.x) + '|' + Math.round(r.y);
                if (seen.has(key)) continue;
                seen.add(key);
                out.push({text: t, x: Math.round(r.x + r.width / 2),
                          y: Math.round(r.y + r.height / 2),
                          w: Math.round(r.width), h: Math.round(r.height)});
            }
            return out;
        }""", text_filter
    )


def find_text(page, target: str, exact: bool = False) -> dict | None:
    """Find the smallest element whose text contains target (or equals it if exact).

    Menu labels carry emoji/newline prefixes (e.g. '🗡️\\nCustom Scenario'), so
    substring matching is the default."""
    els = _scan_elements(page)
    cands = [e for e in els if (e["text"] == target if exact else target in e["text"])]
    if not cands:
        return None
    cands.sort(key=lambda e: e["w"] * e["h"])  # smallest = leaf
    return cands[0]


def find_play(page) -> dict | None:
    """Strict Play matcher: strip emojis, exact-match 'Play'."""
    els = _scan_elements(page)
    for e in els:
        t = e["text"].replace("\u2694\ufe0f", "").replace("\U0001f5e1\ufe0f", "").strip()
        if t == "Play":
            return e
    return None


def is_in_lobby(page) -> bool:
    """Top-left region OCR shows 'Lobby:'/'MP:' -> we are in the multiplayer lobby."""
    try:
        import numpy as np
        from bot.calibration import _ocr_words
        img = np.array(page.screenshot())
        words = _ocr_words(img)
        joined = " ".join(w[0] for w in words)
        return ("Lobby:" in joined) or ("MP:" in joined and "Player Count" in joined)
    except Exception:
        return False


def open_editor(page, log=print, max_attempts: int = 3) -> bool:
    """Open the Custom Scenario editor and VERIFY we got there.
    Menu items are often <div>/<span>, so we scan all elements; if the click
    misses we fall back and retry (Escape first)."""
    for attempt in range(1, max_attempts + 1):
        cs = find_text(page, "Custom Scenario")
        if cs:
            page.mouse.click(cs["x"], cs["y"])
            log(f"[nav] attempt {attempt}: clicked Custom Scenario ({cs['x']},{cs['y']})")
        else:
            page.mouse.click(714, 411)
            log(f"[nav] attempt {attempt}: Custom Scenario fallback coords")
        import time
        time.sleep(3.5)
        if find_text(page, "Back") or find_text(page, "Reset Scenario"):
            return True
        page.keyboard.press("Escape")
        time.sleep(1.5)
    return False


def enter_custom_match(page, log=print, reset_scenario: bool = False) -> bool:
    """Full menu flow: open editor -> (optional reset) -> strict Play -> verify
    not in lobby. Returns True when confirmed inside a custom-scenario match.

    NOTE (empirically verified 2026-08-07): clicking "Reset Scenario" in the
    editor CLOSES the editor and returns to the main menu — so we do NOT click
    it. The editor opens with the correct defaults already (Battle Royale,
    Colors: Random, Uniform: Very Easy, Spawning: Random).
    """
    import time

    if not open_editor(page, log=log):
        log("FATAL: could not open Custom Scenario editor")
        return False

    if reset_scenario:
        reset = find_text(page, "Reset Scenario")
        if reset:
            page.mouse.click(reset["x"], reset["y"])
            log("[nav] clicked Reset Scenario (editor closes to main menu — "
                "reopening)")
            time.sleep(2)
            if not open_editor(page, log=log):
                log("FATAL: could not reopen editor after reset")
                return False

    play = find_play(page)
    if not play:
        page.keyboard.press("Escape")
        time.sleep(1.5)
        play = find_play(page)
    if not play:
        log("NO Play button — aborting")
        return False
    page.mouse.click(play["x"], play["y"])
    log(f"[nav] clicked Play ({play['x']},{play['y']})")
    time.sleep(4)

    # verify we are NOT in the multiplayer lobby; retry if we are
    retries = 0
    while is_in_lobby(page) and retries < 3:
        retries += 1
        log(f"[nav] ERROR: in multiplayer lobby — retry {retries}/3")
        page.keyboard.press("Escape")
        time.sleep(2)
        if not open_editor(page, log=log):
            break
        if reset_scenario:
            reset = find_text(page, "Reset Scenario")
            if reset:
                page.mouse.click(reset["x"], reset["y"])
                time.sleep(3)
        play = find_play(page)
        if play:
            page.mouse.click(play["x"], play["y"])
            time.sleep(4)
    if is_in_lobby(page):
        log("FATAL: still in multiplayer lobby after retries")
        return False
    log("[nav] confirmed: in a custom scenario match (not the lobby)")
    return True
