"""TerritoryNet — small CNN that sees the screen and decides.

Architecture (all from raw screen pixels — genuine vision, no hand-coded rules):
  input   : (B, 3, 64, 64) RGB screen (downsampled)
  encoder : 4 conv blocks -> (B, 64, 16, 16) feature map
  heads:
    - segmentation : 1x1 conv -> (B, 5, 16, 16)  classes: water/neutral/me/enemy/ui
    - localization : pooled -> (B, 2)            my normalized centroid
    - policy       : click-map logits (B, 16*16), action-kind (B,3), attack % (B,1)
    - value        : (B, 1)                      expected return (for RL)
  context (appended to policy/value): my_area_frac, enemy_area_frac, red flag

~100k params; CPU-friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5  # 0 water, 1 neutral, 2 me, 3 enemy, 4 ui


class TerritoryNet(nn.Module):
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

        # segmentation head: per-pixel class logits at grid resolution
        self.seg_head = nn.Conv2d(64, NUM_CLASSES, 1)

        # localization: my centroid (normalized 0..1)
        self.localize_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2),
        )

        # policy heads (share pooled features + context)
        feat_dim = 64 + context_dim
        self.click_head = nn.Sequential(nn.Conv2d(64, 32, 1), nn.ReLU(),
                                        nn.Conv2d(32, 1, 1))  # (B,1,grid,grid) logits
        self.kind_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),  # expand / attack / bank
        )
        self.kind_ctx = nn.Linear(feat_dim, 32)
        self.pct_head = nn.Sequential(nn.Linear(feat_dim + 3, 32), nn.ReLU(), nn.Linear(32, 1))  # sigmoid
        self.value_head = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, rgb, ctx=None, return_all=False):
        """rgb: (B,3,64,64) in 0..1. ctx: (B, context_dim) or None."""
        f = self.encoder(rgb)  # (B,64,16,16)

        seg = self.seg_head(f)  # (B,5,16,16)
        local = self.localize_head(f)  # (B,2)

        # pooled features for policy/value
        pool = F.adaptive_avg_pool2d(f, 1).flatten(1)  # (B,64)
        if ctx is None:
            ctx = torch.zeros(pool.shape[0], self.context_dim, device=rgb.device)
        feat = torch.cat([pool, ctx], dim=1)  # (B, 64+ctx)

        click = self.click_head(f).squeeze(1).flatten(1)  # (B, 256) logits
        kind_logits = self.kind_head(f) + self.kind_ctx(feat)[:, :3]  # (B,3)
        pct_in = torch.cat([feat, torch.softmax(kind_logits, dim=1)], dim=1)
        pct = torch.sigmoid(self.pct_head(pct_in)).squeeze(-1)  # (B,)
        value = self.value_head(feat).squeeze(-1)  # (B,)

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


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
