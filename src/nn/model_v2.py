"""TerritoryNet v2 — the locked blueprint (user-approved 2026-08-09).

Design (hybrid, from the deep discussion):
  * shared dense stem over (rgb + frame-diff) at 128x128
  * pixel branch  -> segmentation (water/neutral/me/enemy/ui) at 16x16
  * global branch -> MoE in the TEACHER only (8 experts, 2 active:
    war / expand / economy specialists) = "many roles per parameter"
  * heads: click-map (16x16), kind (3), pct (1), value (1)
  * numeric context (leaderboard math) injected into the global branch

  Teacher ~100M total / ~40-45M active (sparse)  -> GPU-trained
  Student ~10M dense                            -> distilled, fights live
  Farmer  = model.py 2M                         -> data engine (unchanged)

Width knobs live in one place so 100M<->60M is a one-line fallback.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5
GRID = 16
NUM_CTX = 8  # my_bal_log, my_frac, red, e1_frac, e2_frac, tick_frac, income_log, kills


class _SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.SiLU(), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))[..., None, None]


class _RB(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.se = _SE(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.silu(self.se(self.c2(F.silu(self.c1(x)))) + self.skip(x))


class _Expert(nn.Module):
    """One specialist: two wide residual blocks."""
    def __init__(self, c):
        super().__init__()
        self.b1 = _RB(c, c)
        self.b2 = _RB(c, c)

    def forward(self, x):
        return self.b2(self.b1(x))


class _MoE(nn.Module):
    """8 experts, top-2 routing, load-balance aux loss kept for training."""
    def __init__(self, c, n_exp=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(c) for _ in range(n_exp)])
        self.gate = nn.Linear(c, n_exp)
        self.n_exp, self.top_k = n_exp, top_k

    def forward(self, x):
        B = x.shape[0]
        g = self.gate(F.adaptive_avg_pool2d(x, 1).flatten(1))
        w, idx = g.topk(self.top_k, dim=1)
        w = torch.softmax(w, dim=1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.n_exp):
                m = (idx[:, k] == e)
                if m.any():
                    out[m] += w[m, k].view(-1, 1, 1, 1) * self.experts[e](x[m])
        return out, (g, idx)


def _stem(c0, c1, c2, c3):
    return nn.Sequential(
        nn.Conv2d(6, c0, 3, stride=2, padding=1), nn.SiLU(),   # 128->64
        _RB(c0, c0),
        nn.Conv2d(c0, c1, 3, stride=2, padding=1), nn.SiLU(),   # 64->32
        _RB(c1, c1),
        nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.SiLU(),   # 32->16
        _RB(c2, c3),
    )


class TeacherV2(nn.Module):
    """~100M total, ~40-45M active."""
    def __init__(self, grid=GRID, ctx_dim=NUM_CTX,
                 px=(48, 96, 160, 192), ex=576, heads=256):
        super().__init__()
        self.grid, self.ctx_dim = grid, ctx_dim
        c0, c1, c2, c3 = px
        self.stem = _stem(c0, c1, c2, c3)
        # pixel branch (dense, per-pixel detail)
        self.pix1 = _RB(c3, c3)
        self.seg_head = nn.Conv2d(c3, NUM_CLASSES, 1)
        # global branch (MoE specialists)
        self.down = nn.Conv2d(c3, ex, 3, stride=2, padding=1)   # 16->8
        self.moe = _MoE(ex)
        self.ctx_emb = nn.Linear(ctx_dim, ex)
        # heads
        pool_dim = ex + c3 + ctx_dim
        self.click_head = nn.Sequential(nn.Conv2d(c3, heads, 1), nn.SiLU(),
                                        nn.Conv2d(heads, 1, 1))
        self.kind_head = nn.Sequential(nn.Linear(pool_dim, heads), nn.SiLU(),
                                       nn.Linear(heads, 3))
        self.pct_head = nn.Sequential(nn.Linear(pool_dim + 3, heads), nn.SiLU(),
                                      nn.Linear(heads, 1))
        self.value_head = nn.Sequential(nn.Linear(pool_dim, heads), nn.SiLU(),
                                        nn.Linear(heads, 1))

    def forward(self, rgb, ctx=None, return_all=False):
        """rgb: (B,6,128,128) = [frame, diff] in 0..1. ctx: (B,8)."""
        B = rgb.shape[0]
        f = self.stem(rgb)                       # (B,c3,16,16)
        fp = self.pix1(f)
        seg = self.seg_head(fp)                  # (B,5,16,16)
        fe, _moe_aux = self.moe(self.down(f))    # (B,ex,8,8)
        if ctx is None:
            ctx = torch.zeros(B, self.ctx_dim, device=rgb.device)
        g = F.adaptive_avg_pool2d(fe, 1).flatten(1)          # (B,ex)
        p = F.adaptive_avg_pool2d(fp, 1).flatten(1)          # (B,c3)
        feat = torch.cat([g, p, ctx], dim=1)                 # (B,pool_dim)
        click = self.click_head(fp).squeeze(1).flatten(1)    # (B,256)
        kind = self.kind_head(feat)
        pct = torch.sigmoid(self.pct_head(
            torch.cat([feat, F.softmax(kind, 1)], 1))).squeeze(-1)
        value = self.value_head(feat).squeeze(-1)
        if return_all:
            return seg, click, kind, pct, value
        return click, kind, pct, value


class StudentV2(nn.Module):
    """~10M dense — distilled fighter, same I/O as teacher."""
    def __init__(self, grid=GRID, ctx_dim=NUM_CTX,
                 px=(64, 128, 320, 480), heads=320):
        super().__init__()
        self.grid, self.ctx_dim = grid, ctx_dim
        c0, c1, c2, c3 = px
        self.stem = _stem(c0, c1, c2, c3)
        self.pix1 = _RB(c3, c3)
        self.seg_head = nn.Conv2d(c3, NUM_CLASSES, 1)
        pool_dim = c3 + ctx_dim
        self.click_head = nn.Sequential(nn.Conv2d(c3, heads, 1), nn.SiLU(),
                                        nn.Conv2d(heads, 1, 1))
        self.kind_head = nn.Sequential(nn.Linear(pool_dim, heads), nn.SiLU(),
                                       nn.Linear(heads, 3))
        self.pct_head = nn.Sequential(nn.Linear(pool_dim + 3, heads), nn.SiLU(),
                                      nn.Linear(heads, 1))
        self.value_head = nn.Sequential(nn.Linear(pool_dim, heads), nn.SiLU(),
                                        nn.Linear(heads, 1))

    def forward(self, rgb, ctx=None, return_all=False):
        B = rgb.shape[0]
        f = self.stem(rgb)
        fp = self.pix1(f)
        seg = self.seg_head(fp)
        if ctx is None:
            ctx = torch.zeros(B, self.ctx_dim, device=rgb.device)
        feat = torch.cat([F.adaptive_avg_pool2d(fp, 1).flatten(1), ctx], 1)
        click = self.click_head(fp).squeeze(1).flatten(1)
        kind = self.kind_head(feat)
        pct = torch.sigmoid(self.pct_head(
            torch.cat([feat, F.softmax(kind, 1)], 1))).squeeze(-1)
        value = self.value_head(feat).squeeze(-1)
        if return_all:
            return seg, click, kind, pct, value
        return click, kind, pct, value


def count_params(net):
    return sum(p.numel() for p in net.parameters())
