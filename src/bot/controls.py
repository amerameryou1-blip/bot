"""Input injection for the REAL territorial.io control scheme (click-based).

From the mechanics doc:
  - camera pan: WASD or arrows (does NOT claim land)
  - expand: CLICK adjacent land (cheap, minimal troop cost)
  - attack: set attack-% slider, hover enemy border, press Space (or click)
  - boat: set slider, hover water/coast, press B (3.125% tax + slider %)
  - slider: W +2%, S −2%, D +0.5%, A −0.5%
  - M: auto-attack weakest neighbor; H: hide UI; P: peace vote

The old arrow-based PlaywrightInput is kept for the legacy strategy brain and
tests; MouseControls is what the notebook uses to actually play.
"""
from __future__ import annotations

import time

import numpy as np


class InputSource:
    def set_direction(self, dx: float, dy: float) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class NullInput(InputSource):
    """Logs headings; used in tests and the offline simulator."""

    def __init__(self) -> None:
        self.history: list[tuple[float, float]] = []
        self.current: tuple[float, float] = (0.0, 0.0)

    def set_direction(self, dx: float, dy: float) -> None:
        self.current = (float(dx), float(dy))
        self.history.append(self.current)

    def release(self) -> None:
        self.current = (0.0, 0.0)

    def close(self) -> None:
        self.history.clear()


class PlaywrightInput(InputSource):
    """Arrow-key control (legacy; drives the camera, not the army)."""

    KEY_BY_DIR = {
        "up": "ArrowUp",
        "down": "ArrowDown",
        "left": "ArrowLeft",
        "right": "ArrowRight",
    }

    def __init__(self, page) -> None:
        self.page = page
        self._held: set[str] = set()

    def set_direction(self, dx: float, dy: float) -> None:
        want: set[str] = set()
        if dx > 0.35:
            want.add("right")
        elif dx < -0.35:
            want.add("left")
        if dy < -0.35:
            want.add("up")
        elif dy > 0.35:
            want.add("down")
        for key in self._held - want:
            self.page.keyboard.up(self.KEY_BY_DIR[key])
        for key in want - self._held:
            self.page.keyboard.down(self.KEY_BY_DIR[key])
        self._held = want

    def release(self) -> None:
        for key in list(self._held):
            self.page.keyboard.up(self.KEY_BY_DIR[key])
        self._held.clear()

    def close(self) -> None:
        self.release()


class MouseControls(InputSource):
    """Click-based controls for the actual game.

    The game works on DOUBLE-CLICK (confirmed by the player): double-click
    neutral land to claim it, double-click an enemy border (with the attack
    slider set) to attack. Slider: W +2% / S −2% / D +0.5% / A −0.5%.

    Keeps a tracked slider state; the first set_slider() resets to 0% with a
    burst of S presses (guaranteed low), then steps to the target with
    W/D (up) or S/A (down). Subsequent adjustments are relative and cheap.
    """

    # Where the slider lives (bottom border). Keys only adjust the slider when
    # the pointer is over it, so we park the mouse there first.
    SLIDER_HOVER = (640, 775)

    def __init__(self, page, key_delay: float = 0.04) -> None:
        self.page = page
        self.key_delay = key_delay
        self._slider: float | None = None  # unknown until first set
        self.last_action: str = "none"

    # -- slider ------------------------------------------------------------

    def _press(self, key: str, times: int) -> None:
        for _ in range(times):
            self.page.keyboard.press(key)
            time.sleep(self.key_delay)

    def _reset_to_zero(self) -> None:
        self.page.mouse.move(*self.SLIDER_HOVER)
        time.sleep(0.1)
        self._press("s", 70)  # −2% × 70 guarantees 0
        self._slider = 0.0

    def set_slider(self, pct: float) -> None:
        pct = max(0.0, min(100.0, pct))
        if self._slider is None:
            self._reset_to_zero()
        diff = pct - self._slider
        coarse = int(round(diff / 2.0))
        rem = diff - coarse * 2.0
        self.page.mouse.move(*self.SLIDER_HOVER)
        time.sleep(0.08)
        if coarse > 0:
            self._press("w", coarse)
        elif coarse < 0:
            self._press("s", -coarse)
        fine = int(round(rem / 0.5))
        if fine > 0:
            self._press("d", fine)
        elif fine < 0:
            self._press("a", -fine)
        self._slider = pct

    # -- actions -----------------------------------------------------------

    def expand(self, x: float, y: float) -> None:
        """Double-click neutral land adjacent to my territory to claim it.

        The game claims land on DOUBLE-click (confirmed by the player).
        """
        self.page.mouse.dblclick(x, y)
        self.last_action = f"expand({x:.0f},{y:.0f})"

    def attack(self, x: float, y: float, pct: float) -> None:
        """Hover enemy border, set slider, double-click (land attack)."""
        self.set_slider(pct)
        self.page.mouse.move(x, y)
        time.sleep(0.05)
        self.page.mouse.dblclick(x, y)
        self.last_action = f"attack({x:.0f},{y:.0f},{pct:.0f}%)"

    def send_boat(self, x: float, y: float, pct: float) -> None:
        self.set_slider(pct)
        self.page.mouse.move(x, y)
        time.sleep(0.05)
        self.page.keyboard.press("b")
        self.last_action = f"boat({x:.0f},{y:.0f},{pct:.0f}%)"

    def toggle_auto_attack(self) -> None:
        self.page.keyboard.press("m")
        self.last_action = "auto-attack"

    def peace_vote(self) -> None:
        self.page.keyboard.press("p")
        self.last_action = "peace-vote"

    # -- InputSource compat (unused in click mode) ---------------------------

    def set_direction(self, dx: float, dy: float) -> None:
        pass  # camera moves are handled by clicking

    def release(self) -> None:
        pass

    def close(self) -> None:
        pass
