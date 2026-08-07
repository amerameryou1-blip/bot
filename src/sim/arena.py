"""A tiny headless arena used to battle-test the brain without a browser.

Not a training environment (that comes later, optionally). It builds synthetic
FrameStates from scenario descriptions so tests can assert the brain makes the
right calls: expand when safe, flee from big enemies, attack small ones, and
avoid expansion near danger.
"""
from __future__ import annotations

import numpy as np

from bot.state import Blob, FrameState


def _disc_mask(h: int, w: int, cy: float, cx: float, radius: float, noise: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = dist <= radius
    if noise > 0:
        rng = np.random.default_rng(0)
        mask &= rng.random((h, w)) > noise
    return mask


def build_state(
    h: int = 240,
    w: int = 320,
    self_center: tuple[float, float] = (120.0, 160.0),
    self_radius: float = 30.0,
    enemies: list[tuple[tuple[float, float], float]] | None = None,
    seed: int = 1,
) -> FrameState:
    """Build a FrameState with a me-blob and optional enemy blobs."""
    labels = np.zeros((h, w), dtype=int)

    self_mask = _disc_mask(h, w, *self_center, self_radius)
    labels[self_mask] = 1

    enemies_list: list[Blob] = []
    if enemies:
        for idx, ((cy, cx), r) in enumerate(enemies, start=2):
            mask = _disc_mask(h, w, cy, cx, r)
            labels[mask] = idx
            coords = np.argwhere(mask)
            enemies_list.append(
                Blob(f"enemy{idx}", mask, len(coords), (float(coords[:, 0].mean()), float(coords[:, 1].mean())))
            )

    me_coords = np.argwhere(self_mask)
    me = Blob("me", self_mask, len(me_coords), (float(me_coords[:, 0].mean()), float(me_coords[:, 1].mean())))

    # Frontiers: self cells adjacent to neutral.
    neutral = labels == 0
    from bot.vision import _find_frontiers, find_attack_targets, find_expand_targets

    frontiers = _find_frontiers(self_mask, neutral, max_samples=120)
    expand_targets = find_expand_targets(self_mask, neutral, max_samples=160)
    attack_targets = find_attack_targets(self_mask, labels >= 2, max_samples=120)

    return FrameState(shape=(h, w), labels=labels, self_blob=me, enemies=enemies_list,
                      frontiers=frontiers, expand_targets=expand_targets,
                      attack_targets=attack_targets)
