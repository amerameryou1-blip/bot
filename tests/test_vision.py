"""Unit tests for the vision module using synthetic frames (no browser)."""
from __future__ import annotations

import numpy as np

import pytest

from bot.config import Palette, PlayerColor
from bot.vision import segment


def make_frame(h: int, w: int, patches: dict[tuple[int, int, int], list[tuple[slice, slice]]]) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for color, rects in patches.items():
        for y_slice, x_slice in rects:
            frame[y_slice, x_slice] = color
    return frame


def test_segment_finds_self_and_enemies() -> None:
    palette = Palette(
        self_color=PlayerColor("me", 230, 80, 80),
        enemy_colors=[PlayerColor("blue", 70, 120, 240)],
        tolerance=55.0,
        downscale=1,
    )
    frame = make_frame(200, 200, {
        (230, 80, 80): [(slice(20, 60), slice(20, 60))],
        (70, 120, 240): [(slice(120, 160), slice(120, 160))],
    })
    state = segment(frame, palette)
    assert state.self_blob is not None
    assert state.self_blob.area > 100
    assert len(state.enemies) == 1
    assert state.enemies[0].label == "blue"
    # Both blobs roughly centered where painted.
    cy, cx = state.self_blob.centroid
    assert 35 < cy < 45 and 35 < cx < 45
    ey, ex = state.enemies[0].centroid
    assert 135 < ey < 145 and 135 < ex < 145


def test_segment_neutral_when_colors_unknown() -> None:
    palette = Palette(downscale=1, tolerance=40.0)
    # Grey frame — nothing matches the palette.
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    state = segment(frame, palette)
    assert state.self_blob is None or state.self_blob.area == 0
    assert state.enemies == []
    assert state.frontiers.shape[0] == 0


def test_segment_downscale_keeps_coordinates() -> None:
    palette = Palette(
        self_color=PlayerColor("me", 10, 200, 10),
        downscale=2,
    )
    frame = make_frame(160, 160, {(10, 200, 10): [(slice(40, 80), slice(40, 80))]})
    state = segment(frame, palette)
    assert state.self_blob is not None
    assert state.self_blob.area > 300  # same ballpark as true area (1600)
    # Centroid within the original painted square (40..80).
    cy, cx = state.self_blob.centroid
    assert 40 <= cy <= 80 and 40 <= cx <= 80


def test_segment_finds_frontier_on_self_border() -> None:
    palette = Palette(
        self_color=PlayerColor("me", 230, 80, 80),
        downscale=1,
    )
    # Square of me in the middle, neutral around it.
    frame = make_frame(200, 200, {(230, 80, 80): [(slice(80, 120), slice(80, 120))]})
    state = segment(frame, palette)
    assert state.frontiers.shape[0] > 0
    # Every frontier tile is a self pixel adjacent to neutral — check a sample.
    for row, col in state.frontiers[:5]:
        assert state.self_mask[row, col]
