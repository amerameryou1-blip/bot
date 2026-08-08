"""TerritoryNet v2 (2M) — the 2026-recipe brain.

What changed vs v1 (85k), all borrowed from the best of 2026:
  * residual conv blocks with SE attention (capacity without fragility)
  * SiLU activation (modern smooth gating)
  * ~2M params: still CPU-friendly (small nets are bandwidth-bound, ~1-3ms/think)
  * old 85k net kept as TerritoryNetSmall = distillation TEACHER (stage_seed)

Interface is UNCHANGED (same heads, same forward signature), so the whole
training/playing stack (train_nn.py, rl_loop.py, bot_brain.py) works as-is.

  input   : (B, 3, 64, 64) RGB
  encoder : stem + 6 residual-SE blocks -> (B, 160, 16, 16)
  heads   : seg (5,16,16) / localize (2) / click-map (256) / kind (3) / pct (1) / value (1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5  # 0 water, 1 neutral, 2 me, 3 enemy, 4 ui


class _SE(nn.Module):
    """Squeeze-Excitation: channel attention, ~0.1% params, real accuracy."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.SiLU(), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x * self.fc(w)[..., None, None]


class _RB(nn.Module):
    """Residual block: conv-SiLU-conv + SE, with 1x1 skip when channels change."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.se = _SE(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = F.silu(self.c1(x))
        y = self.se(self.c2(y))
        return F.silu(y + self.skip(x))


class TerritoryNet(nn.Module):
    """~2M param brain (v2). Same API as v1."""

    CH = (48, 96, 160, 160, 160, 160)

    def __init__(self, grid: int = 16, context_dim: int = 3):
        super().__init__()
        self.grid = grid
        self.context_dim = context_dim
        c0, c1, c2, c3, c4, c5 = self.CH

        self.encoder = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1), nn.SiLU(),      # 64x64
            _RB(c0, c0),
            nn.MaxPool2d(2), _RB(c0, c1),                    # 32x32
            nn.MaxPool2d(2), _RB(c1, c2),                    # 16x16
            _RB(c2, c3), _RB(c3, c4), _RB(c4, c5),
        )
        C = c5

        # segmentation head: per-pixel class logits at grid resolution
        self.seg_head = nn.Conv2d(C, NUM_CLASSES, 1)

        # localization: my centroid (normalized 0..1)
        self.localize_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, 64), nn.SiLU(), nn.Linear(64, 2),
        )

        # policy heads (share pooled features + context)
        feat_dim = C + context_dim
        self.click_head = nn.Sequential(nn.Conv2d(C, 64, 1), nn.SiLU(),
                                        nn.Conv2d(64, 1, 1))  # (B,1,grid,grid)
        self.kind_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, 64), nn.SiLU(),
            nn.Linear(64, 3),  # expand / attack / bank
        )
        self.kind_ctx = nn.Linear(feat_dim, 64)
        self.pct_head = nn.Sequential(nn.Linear(feat_dim + 3, 64), nn.SiLU(), nn.Linear(64, 1))
        self.value_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, rgb, ctx=None, return_all=False):
        """rgb: (B,3,64,64) in 0..1. ctx: (B, context_dim) or None."""
        f = self.encoder(rgb)  # (B,C,16,16)

        seg = self.seg_head(f)
        local = self.localize_head(f)

        pool = F.adaptive_avg_pool2d(f, 1).flatten(1)
        if ctx is None:
            ctx = torch.zeros(pool.shape[0], self.context_dim, device=rgb.device)
        feat = torch.cat([pool, ctx], dim=1)

        click = self.click_head(f).squeeze(1).flatten(1)  # (B, 256)
        kind_logits = self.kind_head(f) + self.kind_ctx(feat)[:, :3]
        pct_in = torch.cat([feat, torch.softmax(kind_logits, dim=1)], dim=1)
        pct = torch.sigmoid(self.pct_head(pct_in)).squeeze(-1)
        value = self.value_head(feat).squeeze(-1)

        if return_all:
            return seg, local, click, kind_logits, pct, value
        return click, kind_logits, pct, value

    # -- helpers ------------------------------------------------------------

    def seg_probs(self, rgb):
        seg, *_ = self.forward(rgb, return_all=True)
        return F.softmax(seg, dim=1)

    def act(self, rgb, ctx=None, greedy=True):
        """Return (kind_index, cell_index, pct, value) for a single obs."""
        self.eval()
        with torch.no_grad():
            seg, local, click, kind, pct, value = self.forward(rgb, ctx, return_all=True)
            if greedy:
                cell = int(click[0].argmax())
                kind_i = int(kind[0].argmax())
            else:
                cell = int(torch.multinomial(F.softmax(click[0], dim=-1), 1))
                kind_i = int(torch.multinomial(F.softmax(kind[0], dim=-1), 1))
            return kind_i, cell, float(pct[0]), float(value[0])


class TerritoryNetSmall(nn.Module):
    """v1 85k brain — kept as distillation TEACHER and to load old weights."""

    def __init__(self, grid: int = 16, context_dim: int = 3):
        super().__init__()
        self.grid = grid
        self.context_dim = context_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.seg_head = nn.Conv2d(64, NUM_CLASSES, 1)
        self.localize_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2),
        )
        feat_dim = 64 + context_dim
        self.click_head = nn.Sequential(nn.Conv2d(64, 32, 1), nn.ReLU(),
                                        nn.Conv2d(32, 1, 1))
        self.kind_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.kind_ctx = nn.Linear(feat_dim, 32)
        self.pct_head = nn.Sequential(nn.Linear(feat_dim + 3, 32), nn.ReLU(), nn.Linear(32, 1))
        self.value_head = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, rgb, ctx=None, return_all=False):
        f = self.encoder(rgb)
        seg = self.seg_head(f)
        local = self.localize_head(f)
        pool = F.adaptive_avg_pool2d(f, 1).flatten(1)
        if ctx is None:
            ctx = torch.zeros(pool.shape[0], self.context_dim, device=rgb.device)
        feat = torch.cat([pool, ctx], dim=1)
        click = self.click_head(f).squeeze(1).flatten(1)
        kind_logits = self.kind_head(f) + self.kind_ctx(feat)[:, :3]
        pct_in = torch.cat([feat, torch.softmax(kind_logits, dim=1)], dim=1)
        pct = torch.sigmoid(self.pct_head(pct_in)).squeeze(-1)
        value = self.value_head(feat).squeeze(-1)
        if return_all:
            return seg, local, click, kind_logits, pct, value
        return click, kind_logits, pct, value


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
