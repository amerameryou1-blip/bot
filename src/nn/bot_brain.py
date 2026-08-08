"""NeuralBrain — the CNN drives the LIVE bot (no palette, no hand-coded rules).

Flow per decision:
  1. screenshot (1280x800) -> 64x64 RGB tensor
  2. net forward (seg + policy)
  3. context (my_frac, enemy_frac, red) from the net's OWN segmentation
  4. policy -> (kind, cell, pct) -> screen pixel -> ClickAction

The segmentation head is the "eyes": water / neutral / me / enemy / UI.
Fallback to the heuristic ClickPlanner if no model is available.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bot.planner import ClickAction
from bot.economy import TroopTracker

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

GRID = 16
SIZE = 64
UI_CLASS = 4


def load_model(path=None):
    """Load the best available brain, shape-safe:
    2M distilled/rl checkpoint -> 2M model.pt -> trained 85k teacher."""
    if not _HAS_TORCH:
        return None
    from nn.model import TerritoryNet, TerritoryNetSmall
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(REPO / "weights" / "nn" / "rl" / "best.pt")
    candidates.append(REPO / "weights" / "nn" / "model.pt")
    for c in candidates:
        if c.exists():
            try:
                net = TerritoryNet(grid=GRID, context_dim=3)
                net.load_state_dict(torch.load(c, map_location="cpu"))
                return net
            except Exception:
                continue
    tp = REPO / "weights" / "nn" / "teacher.pt"
    if tp.exists():
        try:
            t = TerritoryNetSmall(grid=GRID, context_dim=3)
            t.load_state_dict(torch.load(tp, map_location="cpu"))
            return t
        except Exception:
            pass
    return None


class NeuralBrain:
    """decide(frame_rgb) -> ClickAction. frame_rgb: (H, W, 3) uint8."""

    def __init__(self, net, tracker: TroopTracker | None = None):
        self.net = net
        self.tracker = tracker or TroopTracker(balance=512.0, land=12)
        self.last_ctx = None

    def _prep(self, frame):
        from PIL import Image
        img = Image.fromarray(frame).resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.tensor(np.array(img).transpose(2, 0, 1)[None], dtype=torch.float32) / 255.0
        return x

    def _seg_context(self, seg_probs, h, w):
        """Context from the net's own segmentation."""
        p = seg_probs[0]  # (5, 16, 16)
        my_frac = float(p[2].mean())
        enemy_frac = float(p[3].mean())
        red = 1.0 if self.tracker.density() >= 90 else 0.0
        return torch.tensor([[my_frac, enemy_frac, red]])

    def decide(self, frame_rgb) -> ClickAction:
        self.net.eval()
        with torch.no_grad():
            x = self._prep(frame_rgb)
            seg, local, click, kind, pct, _ = self.net.forward(x, None, return_all=True)
            ctx = self._seg_context(F.softmax(seg, dim=1), *frame_rgb.shape[:2])
            seg, local, click, kind, pct, _ = self.net.forward(x, ctx, return_all=True)
            kind_i = int(kind[0].argmax())
            if kind_i == 2:
                return ClickAction("bank", reason="nn-bank")
            cell = int(click[0].argmax())
            cy, cx = divmod(cell, GRID)
            h, w = frame_rgb.shape[:2]
            y = (cy + 0.5) / GRID * h
            xp = (cx + 0.5) / GRID * w
            kind_s = {0: "expand", 1: "attack"}[kind_i]
            pct_v = float(pct[0]) * 100.0 if kind_i == 1 else 0.0
            return ClickAction(kind_s, float(xp), float(y), pct_v, reason=f"nn-{kind_s}")

    def segment_map(self, frame_rgb) -> np.ndarray:
        """Return the net's per-pixel class map at 64x64 (useful for logging)."""
        self.net.eval()
        with torch.no_grad():
            x = self._prep(frame_rgb)
            seg, *_ = self.net.forward(x, None, return_all=True)
            return int(F.softmax(seg, dim=1)[0].argmax(dim=0))
