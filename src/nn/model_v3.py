"""TerritoryNet TEACHER v3 (2026-08-14, self-improvement pass vs v2).

Fixes every weakness I listed for v2 (and handed to the rival as ammo):
  * MEMORY: 2-frame stack (current+previous @128) instead of frame+diff.
  * SHARP ACTIONS: native 32x32 click head (1024 cells) — shards record cell
    on 32x32, v2 was downsampling the signal to 16x16.
  * SHARP VISION: segmentation head at 32x32 (4 classes), not 16x16.
  * MoE KEPT (blueprint) but load-balance aux loss is now RETURNED and used,
    so experts cannot collapse.
  * Advantage-weighted BC lives in train_v2 (weights from episode returns),
    so the teacher learns mostly from the farmer's GOOD episodes.
Size target ~90-100M total, ~35M active (2/8 experts) — fits T4x2 PPO.
Contract (same as v2 so the pipeline/eval speak it):
  forward(rgb (B,6,128,128) [cur+prev], ctx (B,8)) ->
  click (B,1024), kind (B,3), pct (B,), value (B,)
  return_all=True also gives seg (B,4,32,32) and moe aux loss.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

SEG_CLASSES = 4          # water / land / me / enemy (sim labels)
GRID = 32                # native action grid of the shards
NUM_CTX = 8


class _SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.SiLU(), nn.Linear(h, c),
                                nn.Sigmoid())

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
    def __init__(self, c):
        super().__init__()
        self.b1 = _RB(c, c)
        self.b2 = _RB(c, c)

    def forward(self, x):
        return self.b2(self.b1(x))


class _MoE(nn.Module):
    """8 experts top-2 + SWITCH-style load balance aux loss (v3: actually used)."""
    def __init__(self, c, n_exp=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(c) for _ in range(n_exp)])
        self.gate = nn.Linear(c, n_exp)
        self.n_exp, self.top_k = n_exp, top_k

    def forward(self, x):
        g = self.gate(F.adaptive_avg_pool2d(x, 1).flatten(1))   # (B, E)
        w, idx = g.topk(self.top_k, dim=1)
        w = torch.softmax(w, dim=1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.n_exp):
                m = (idx[:, k] == e)
                if m.any():
                    out[m] += w[m, k].view(-1, 1, 1, 1) * self.experts[e](x[m])
        # load balance: uniform target over experts for top-1 mass
        p = torch.softmax(g, dim=1)
        freq = torch.stack([(idx[:, 0] == e).float().mean()
                            for e in range(self.n_exp)])
        aux = (p.mean(0) * freq).sum() * self.n_exp   # min at uniform
        return out, aux


class TeacherV3(nn.Module):
    def __init__(self, grid=GRID, ctx_dim=NUM_CTX,
                 px=(64, 128, 256, 512), ex=512, heads=256):
        super().__init__()
        self.grid, self.ctx_dim = grid, ctx_dim
        c0, c1, c2, c3 = px
        self.stem = nn.Sequential(
            nn.Conv2d(6, c0, 3, stride=2, padding=1), nn.SiLU(),   # 128->64
            _RB(c0, c0),
            nn.Conv2d(c0, c1, 3, stride=2, padding=1), nn.SiLU(),  # 64->32
            _RB(c1, c1),
            _RB(c1, c2),
            _RB(c2, c3),
        )
        self.pix1 = _RB(c3, c3)
        self.seg_head = nn.Conv2d(c3, SEG_CLASSES, 1)
        self.down = nn.Conv2d(c3, ex, 3, stride=2, padding=1)       # 32->16
        self.moe = _MoE(ex)
        self.ctx_emb = nn.Linear(ctx_dim, ex)
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
        B = rgb.shape[0]
        f = self.stem(rgb)                              # (B,c3,32,32)
        fp = self.pix1(f)
        seg = self.seg_head(fp)                         # (B,4,32,32)
        fe, aux = self.moe(self.down(f))                # (B,ex,16,16)
        if ctx is None:
            ctx = torch.zeros(B, self.ctx_dim, device=rgb.device)
        g = F.adaptive_avg_pool2d(fe, 1).flatten(1)
        p = F.adaptive_avg_pool2d(fp, 1).flatten(1)
        feat = torch.cat([g, p, ctx], dim=1)
        click = self.click_head(fp).squeeze(1).flatten(1)          # (B,1024)
        kind = self.kind_head(feat)
        pct = torch.sigmoid(self.pct_head(
            torch.cat([feat, F.softmax(kind, 1)], 1))).squeeze(-1)
        value = self.value_head(feat).squeeze(-1)
        if return_all:
            return seg, click, kind, pct, value, aux
        return click, kind, pct, value


def count_params(net):
    return sum(p.numel() for p in net.parameters())
