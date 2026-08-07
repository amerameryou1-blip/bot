"""The main loop: capture -> segment -> decide -> act, on a fixed cadence."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import LoopConfig, Palette
from .controls import InputSource
from .strategy import Priority, StrategyBrain
from .vision import segment


@dataclass
class LoopStats:
    ticks: int = 0
    frames_captured: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    max_self_area: int = 0
    total_think_ms: float = 0.0

    def record(self, decision) -> None:
        self.ticks += 1
        self.reason_counts[decision.reason] = self.reason_counts.get(decision.reason, 0) + 1

    def snapshot(self) -> dict:
        return {
            "ticks": self.ticks,
            "frames_captured": self.frames_captured,
            "max_self_area": self.max_self_area,
            "reasons": dict(sorted(self.reason_counts.items(), key=lambda kv: -kv[1])[:8]),
            "avg_think_ms": round(self.total_think_ms / max(self.ticks, 1), 2),
        }


class BotLoop:
    """Wires capture (callable -> np.ndarray) + brain + input together."""

    def __init__(
        self,
        capture,
        palette: Palette,
        brain: StrategyBrain,
        inputs: InputSource,
        loop_cfg: LoopConfig | None = None,
        log=print,
    ) -> None:
        self.capture = capture
        self.palette = palette
        self.brain = brain
        self.inputs = inputs
        self.cfg = loop_cfg or LoopConfig()
        self.log = log
        self.stats = LoopStats()
        self._last_heading = np.array([0.0, 0.0])
        self._tick_ms = 1000.0 / self.cfg.hz

    def run(self, duration_s: float | None = None, max_ticks: int | None = None) -> LoopStats:
        started = time.perf_counter()
        tick = 0
        while True:
            tick += 1
            self._tick_once()
            if max_ticks is not None and tick >= max_ticks:
                break
            if duration_s is not None and (time.perf_counter() - started) >= duration_s:
                break
            # Throttle to the configured cadence.
            elapsed = time.perf_counter() - started - tick * self._tick_ms / 1000.0
            if self._tick_ms > 0:
                dt = (tick * self._tick_ms / 1000.0) - (time.perf_counter() - started)
                if dt > 0:
                    time.sleep(dt)
        self.inputs.release()
        return self.stats

    def _tick_once(self) -> None:
        t0 = time.perf_counter()
        frame = self.capture()
        self.stats.frames_captured += 1

        state = segment(frame, self.palette)
        decision = self.brain.decide(state, self._last_heading)
        self._last_heading = np.array([decision.dx, decision.dy])

        self.inputs.set_direction(decision.dx, decision.dy)

        think_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.total_think_ms += think_ms
        self.stats.record(decision)
        if state.self_blob:
            self.stats.max_self_area = max(self.stats.max_self_area, state.self_blob.area)

        if self.cfg.log_every > 0 and self.stats.ticks % self.cfg.log_every == 0:
            self.log(f"[{self.stats.ticks:>5}] {state.summary()} | {decision.reason} | {think_ms:.0f}ms")

        if self.cfg.stats_every > 0 and self.stats.ticks % self.cfg.stats_every == 0:
            self.log(f"STATS {self.stats.snapshot()}")
