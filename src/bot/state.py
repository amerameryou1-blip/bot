"""Data model for what the bot "sees" each frame."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Blob:
    """A connected-ish colored region: its territory."""
    label: str          # 'me' or enemy color name
    mask: np.ndarray    # (H, W) bool
    area: int           # pixel count
    centroid: tuple[float, float]  # (row, col)

    @property
    def radius(self) -> float:
        """Approximate radius of the blob, assuming roughly circular."""
        return float(np.sqrt(self.area / np.pi))


@dataclass
class FrameState:
    """Everything the strategy brain needs, derived from one screenshot."""
    shape: tuple[int, int]                     # (H, W)
    labels: np.ndarray                         # (H, W) int: 0 neutral, 1 me, 2+ enemies
    self_blob: Blob | None
    enemies: list[Blob] = field(default_factory=list)
    # Candidate expansion tiles: points on MY frontier adjacent to neutral.
    frontiers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    # NEUTRAL pixels adjacent to my land — click these to expand (click-based mode).
    expand_targets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    # ENEMY pixels adjacent to my land — hover + Space to attack.
    attack_targets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))

    @property
    def neutral_mask(self) -> np.ndarray:
        return self.labels == 0

    @property
    def self_mask(self) -> np.ndarray:
        return self.labels == 1

    def summary(self) -> str:
        me = f"me(area={self.self_blob.area if self.self_blob else 0})"
        foes = ", ".join(f"{e.label}({e.area})" for e in self.enemies[:4])
        return f"{me} | enemies: {foes or 'none'} | expand:{len(self.expand_targets)} attack:{len(self.attack_targets)}"
