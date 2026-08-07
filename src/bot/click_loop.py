"""Click-based game loop — the real control scheme.

capture -> segment -> ClickPlanner.decide -> MouseControls.execute
at a modest cadence (the game processes clicks at conquest mini-tick speed;
spamming clicks is wasteful). Leaderboard OCR of balance/land can be fed in
periodically via `brain.set_observed_balance`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import LoopConfig, Palette
from .controls import MouseControls
from .planner import ClickPlanner
from .vision import segment


@dataclass
class ClickStats:
    ticks: int = 0
    frames: int = 0
    actions: dict[str, int] = field(default_factory=dict)
    max_area: int = 0
    total_think_ms: float = 0.0

    def record(self, action) -> None:
        self.ticks += 1
        key = action.kind
        self.actions[key] = self.actions.get(key, 0) + 1

    def snapshot(self) -> dict:
        return {
            "ticks": self.ticks,
            "frames": self.frames,
            "max_area": self.max_area,
            "actions": dict(sorted(self.actions.items(), key=lambda kv: -kv[1])),
            "avg_think_ms": round(self.total_think_ms / max(self.ticks, 1), 2),
        }


class ClickLoop:
    def __init__(
        self,
        capture,
        palette: Palette,
        brain: ClickPlanner,
        controls: MouseControls,
        loop_cfg: LoopConfig | None = None,
        log=print,
        decision_interval_s: float = 0.4,
    ) -> None:
        self.capture = capture
        self.palette = palette
        self.brain = brain
        self.controls = controls
        self.cfg = loop_cfg or LoopConfig()
        self.log = log
        self.stats = ClickStats()
        self.decision_interval = decision_interval_s
        self._last_decision = 0.0

    def run(self, duration_s: float | None = None, max_ticks: int | None = None) -> ClickStats:
        started = time.perf_counter()
        ticks = 0
        while True:
            ticks += 1
            self._tick_once()
            if max_ticks is not None and ticks >= max_ticks:
                break
            if duration_s is not None and (time.perf_counter() - started) >= duration_s:
                break
            time.sleep(max(0.0, self.decision_interval - (time.perf_counter() - started - ticks * self.decision_interval)))
        return self.stats

    def _tick_once(self) -> None:
        t0 = time.perf_counter()
        frame = self.capture()
        self.stats.frames += 1

        state = segment(frame, self.palette)
        action = self.brain.decide(state)
        self.stats.record(action)

        # execute the action
        try:
            if action.kind == "expand" and action.x > 0:
                self.controls.expand(action.x, action.y)
            elif action.kind == "attack" and action.x > 0:
                self.controls.attack(action.x, action.y, action.pct)
            elif action.kind == "boat" and action.x > 0:
                self.controls.send_boat(action.x, action.y, action.pct)
            elif action.kind == "peace":
                self.controls.peace_vote()
            # 'bank' → no input (compound), by design
        except Exception as e:  # never let one bad click kill the match
            self.log(f"  control error: {e}")

        if state.self_blob:
            self.stats.max_area = max(self.stats.max_area, state.self_blob.area)

        think_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.total_think_ms += think_ms

        if self.cfg.log_every > 0 and self.stats.ticks % self.cfg.log_every == 0:
            self.log(f"[{self.stats.ticks:>4}] {state.summary()} | {action.reason_label} | {think_ms:.0f}ms")
        if self.cfg.stats_every > 0 and self.stats.ticks % self.cfg.stats_every == 0:
            self.log(f"STATS {self.stats.snapshot()}")
