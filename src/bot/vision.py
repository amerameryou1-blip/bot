"""Fast color-based vision: screenshot -> FrameState.

No heavy model needed. The game paints each player's territory in one color,
so nearest-color segmentation is both faster and more reliable than a neural
net for this task. Everything is vectorized numpy and runs in milliseconds.
"""
from __future__ import annotations

import numpy as np

from .config import Palette
from .state import Blob, FrameState


def _nearest_color_labels(
    frame: np.ndarray,  # (H, W, 3) uint8 RGB
    palette: Palette,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each pixel by nearest palette color; return labels + a validity mask.

    labels: (H, W) int, 0..K-1 (palette index)
    valid:  (H, W) bool, True where the nearest color is close enough to count.
    """
    colors = np.array([c.rgb for c in palette.all_colors], dtype=np.float32)  # (K, 3)
    pix = frame.astype(np.float32)
    # Squared distance to every palette color, vectorized.
    diff = pix[:, :, None, :] - colors[None, None, :, :]
    dist2 = np.einsum("hwkc->hwk", diff * diff)  # (H, W, K)
    labels = dist2.argmin(axis=-1)
    nearest = dist2.min(axis=-1)
    valid = nearest <= palette.tolerance ** 2
    return labels, valid


def _blob_from_mask(mask: np.ndarray, label: str) -> Blob:
    coords = np.argwhere(mask)
    area = len(coords)
    if area == 0:
        return Blob(label, mask, 0, (0.0, 0.0))
    centroid = coords.mean(axis=0)
    return Blob(label, mask, area, (float(centroid[0]), float(centroid[1])))


def _find_frontiers(self_mask: np.ndarray, neutral_mask: np.ndarray, max_samples: int = 120) -> np.ndarray:
    """Points on my territory's border adjacent to neutral land."""
    # Simple 4-neighbor check: a cell of mine with a neutral neighbor is a frontier.
    padded = np.pad(self_mask, 1)
    touched_by_neutral = (
        np.pad(neutral_mask, 1)[1:-1, :-2] |   # left neighbor
        np.pad(neutral_mask, 1)[1:-1, 2:] |    # right neighbor
        np.pad(neutral_mask, 1)[:-2, 1:-1] |   # up neighbor
        np.pad(neutral_mask, 1)[2:, 1:-1]      # down neighbor
    )
    frontier = self_mask & touched_by_neutral
    coords = np.argwhere(frontier)
    if len(coords) <= max_samples:
        return coords
    # Sample evenly so the brain sees the whole border, not just the top.
    idx = np.linspace(0, len(coords) - 1, max_samples, dtype=int)
    return coords[idx]


def _adjacent_mask(mask: np.ndarray) -> np.ndarray:
    """4-neighbour dilation: True where adjacent to `mask`."""
    padded = np.pad(mask, 1)
    return (
        padded[1:-1, :-2] | padded[1:-1, 2:] |
        padded[:-2, 1:-1] | padded[2:, 1:-1]
    )


def ui_mask(h: int, w: int) -> np.ndarray:
    """Screen rects that are UI, not map (1280x800 layout, scaled). Clicks
    there opened modals / did nothing — the match-#6 disaster. Data audit
    2026-08-09: up to 95% of early clicks were UI garbage."""
    sx, sy = w / 1280.0, h / 800.0
    m = np.zeros((h, w), dtype=bool)
    m[: int(50 * sy), :] = True                      # top banner
    m[: int(320 * sy), : int(300 * sx)] = True       # leaderboard
    m[int(740 * sy):, :] = True                      # bottom bar
    m[:, int(1210 * sx):] = True                     # zoom buttons column
    return m


def find_expand_targets(self_mask: np.ndarray, neutral_mask: np.ndarray, max_samples: int = 120) -> np.ndarray:
    """NEUTRAL pixels adjacent to my territory — these are what I click to claim."""
    targets = neutral_mask & _adjacent_mask(self_mask)
    coords = np.argwhere(targets)
    if len(coords) <= max_samples:
        return coords
    idx = np.linspace(0, len(coords) - 1, max_samples, dtype=int)
    return coords[idx]


def find_attack_targets(self_mask: np.ndarray, enemy_mask: np.ndarray, max_samples: int = 120) -> np.ndarray:
    """ENEMY pixels adjacent to my territory — hover + Space to attack."""
    targets = enemy_mask & _adjacent_mask(self_mask)
    coords = np.argwhere(targets)
    if len(coords) <= max_samples:
        return coords
    idx = np.linspace(0, len(coords) - 1, max_samples, dtype=int)
    return coords[idx]


def segment(frame: np.ndarray, palette: Palette) -> FrameState:
    """Convert one screenshot (RGB, uint8) into a FrameState.

    Input can be any size; it's downsampled for speed then results are scaled
    back to the input coordinate space.
    """
    ds = max(1, palette.downscale)
    small = frame[::ds, ::ds]

    labels_small, valid_small = _nearest_color_labels(small, palette)

    # Remap palette indices -> frame labels: neutral/invalid = 0, me = 1,
    # enemies = 2..K. (Palette index 0 is the self color, so +1 shifts all.)
    labels = np.repeat(np.repeat(labels_small + 1, ds, axis=0), ds, axis=1)
    valid = np.repeat(np.repeat(valid_small, ds, axis=0), ds, axis=1)
    labels = np.where(valid, labels, 0)

    h, w = labels.shape
    um = ui_mask(h, w)
    neutral_mask = (labels == 0) & ~um

    # Self blob: label 1..n_self (self + lightened aliases).
    n_self = 1 + len(getattr(palette, "self_aliases", []))
    self_mask = (labels >= 1) & (labels <= n_self)
    self_blob = _blob_from_mask(self_mask, "me")

    # Enemy blobs: labels after the self aliases, never inside UI rects.
    enemies: list[Blob] = []
    for idx in range(n_self + 1, len(palette.all_colors) + 1):
        mask = (labels == idx) & ~um
        if mask.any():
            color = palette.all_colors[idx - 1]
            bl = _blob_from_mask(mask, color.name)
            cy, cx = bl.centroid
            if not um[int(cy), int(cx)]:
                enemies.append(bl)

    frontiers = _find_frontiers(self_mask, neutral_mask) if self_blob.area > 0 else np.zeros((0, 2), dtype=int)

    enemy_mask = (labels >= 2) & ~um
    expand_targets = (
        find_expand_targets(self_mask, neutral_mask, max_samples=160)
        if self_blob.area > 0 else np.zeros((0, 2), dtype=int)
    )
    attack_targets = (
        find_attack_targets(self_mask, enemy_mask, max_samples=120)
        if self_blob.area > 0 else np.zeros((0, 2), dtype=int)
    )

    return FrameState(shape=(h, w), labels=labels, self_blob=self_blob, enemies=enemies,
                      frontiers=frontiers, expand_targets=expand_targets,
                      attack_targets=attack_targets)
