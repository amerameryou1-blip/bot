"""Tests for per-match color calibration (leaderboard swatch reading)."""
from __future__ import annotations

import numpy as np

from bot.calibration import (
    blob,
    edges_touched,
    saturated_colors,
    swatch_from_strip,
    validate_territory_color,
)


def _img_with(swatch_rgb, territory_rgb):
    """Synthetic 800x1200 frame: dark panel + swatch square + a territory blob
    outside the leaderboard corner."""
    img = np.zeros((800, 1200, 3), dtype=np.uint8)
    img[:] = (8, 40, 20)  # dark panel-ish bg
    img[10:30, 10:30] = swatch_rgb  # leaderboard swatch, top-left
    yy, xx = np.ogrid[:800, :1200]
    disc = (yy - 500) ** 2 + (xx - 700) ** 2 < 60 ** 2  # territory, bottom-right
    img[disc] = territory_rgb
    return img


def test_swatch_from_strip_picks_saturated_color():
    # strip: mostly dark panel bg + a small saturated square
    strip = np.zeros((20, 60, 3), dtype=np.uint8)
    strip[:] = (6, 40, 18)
    strip[5:15, 30:38] = (242, 216, 63)  # yellow swatch
    color = swatch_from_strip(strip.reshape(-1, 3))
    assert color is not None
    r, g, b = color
    assert g > 150 and r > 150 and b < 120  # yellow-ish


def test_swatch_from_strip_rejects_dull_strip():
    strip = np.full((20, 60, 3), (6, 40, 18), dtype=np.uint8)
    assert swatch_from_strip(strip.reshape(-1, 3)) is None


def test_edges_touched_counts_borders():
    mask = np.zeros((100, 100), dtype=bool)
    mask[0, 10:90] = True  # top edge only (not the corners)
    assert edges_touched(mask) == 1
    mask[10:90, 0] = True  # left edge
    assert edges_touched(mask) == 2
    mask[0:100, 99] = True  # right edge
    mask[99, 0:100] = True  # bottom edge
    assert edges_touched(mask) == 4


def test_validate_territory_color_accepts_real_blob():
    img = _img_with((242, 216, 63), (242, 216, 63))
    assert validate_territory_color(img, (240, 216, 64))


def test_validate_territory_color_rejects_background():
    img = _img_with((242, 216, 63), (40, 200, 40))
    # swatch color only exists as the tiny top-left square -> rejected
    assert not validate_territory_color(img, (240, 216, 64), tol=48)
    # a full-frame color -> touches all edges
    big = np.zeros((800, 1200, 3), dtype=np.uint8)
    big[:] = (5, 90, 16)
    assert not validate_territory_color(big, (5, 90, 16))


def test_saturated_colors_finds_territories():
    img = _img_with((242, 216, 63), (40, 200, 40))
    colors = saturated_colors(img, max_colors=8)
    rgb = [tuple(c) for c in colors]
    assert (240, 216, 48) in rgb or (240, 216, 72) in rgb or (232, 216, 48) in rgb or (232, 216, 72) in rgb


def test_blob_returns_stats():
    img = _img_with((242, 216, 63), (40, 200, 40))
    b = blob(img, (40, 200, 40), tol=48)
    assert b is not None
    assert b["area"] > 1000
    assert 600 < b["cx"] < 800 and 400 < b["cy"] < 600
