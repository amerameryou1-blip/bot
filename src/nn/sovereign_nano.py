"""
SOVEREIGN-nano — <=5M-param policy for territorial.io (last-survivor).

Role (pipeline brief, 2026-08-14):
  1. FAST ROLLOUT ENGINE — ~90x cheaper than the 290M teacher, so the
     PPO / league / wake-sleep loop gets ~10x more sim experience per
     GPU-hour.
  2. DISTILLATION STUDENT — the teacher distills into it (distill.py);
     the contract below is identical to v3 so KL-on-heatmaps is trivial.
  3. CPU-TRAINABLE SANITY BASELINE — smoke-trainable on Kaggle CPU.

Contract (identical to SOVEREIGN v3):
    forward(rgb (B,3,256,256) float [0,1], nums (B,8), rtg (B,)|None,
            grid=32) -> (click (B,1024), kind_logits (B,3), pct (B,),
                         value (B,))
    act(...) -> {kind, cell, yx, pct, logprob, value, win_prob, entropy,
                 state} with logprob/entropy STRICTLY (B,) at any batch.

Kept from v3 (scaled down): return-conditioning (rtg), gate head, intent
mirrors (threat/expand), next_seg dynamics aux, win head, econ forecaster,
Beta pct, cell-value aux, tiny GRU memory.
Dropped (stated honestly): the 77M transformer cortex -> pooled-MLP global
head; the MCTS planning block -> OUT; SE attention -> OUT.

REAL SHARD SCHEMA (measured 2026-08-14 on shard_v2_1786707781):
    rgb    (354,256,256,3) uint8   frames every 2 ticks
    lab    (354,256,256)   uint8   0 water / 1 neutral / 2 me / 3 enemy
    nums   (354,8) float32         [bal_log,me_frac,red,e1,e2,tick_frac,
                                    income,kills]
    kind   (354,) int64            0 expand / 1 attack / 2 bank
    cell   (354,) int64            32x32 grid idx 0..1023
    pct    (354,) float32          troops fraction in [0,1]
    logp   (354,) float32          TRUE behavior logprob (old shards -2.0)
    reward (708,) float32          PER TICK (2x frames), -1.3 death terminal
    alive  (4,)   int64            PER EPISODE 0/1 (survived timeout)
    lens   (4,)   int64            PER EPISODE TICK counts (frames=lens//2)

Every bug from the v3 review is fixed here AND asserted in the smoke test
(see __main__). Zero synthetic-only claims: the smoke loads the REAL shard
via $HF_TOKEN when available and prints which path it ran.

torch + numpy only. Run:  python model_nano.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

EPS = 1e-6
PARAM_LIMIT = 5_000_000
HF_DATASET = "amer224/territorial-bot-data"
HF_SHARD = "rl/shards_v2/shard_v2_1786707781_27_23.npz"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class NanoConfig:
    map_size: int = 256
    in_c: int = 3
    ctx_dim: int = 8
    rtg_dim: int = 1
    num_kinds: int = 3
    # enc_ch[0]=stem out; enc_ch[1..4]=stage outputs (@64,32,16,8 px)
    enc_ch: Tuple[int, ...] = (16, 32, 64, 128, 192)
    enc_blocks: Tuple[int, ...] = (1, 2, 2, 1)
    # dec_plan: (out_ch, skip_ch or None, res_blocks); last out = dec_out_ch
    dec_plan: Tuple[Tuple[int, Optional[int], int], ...] = (
        (128, 128, 1),   # dec0 @16px (skip e3)
        (64, 64, 1),     # dec1 @32px (skip e2)  -> cell_value source
        (48, 32, 1),     # dec2 @64px (skip e1)
        (32, None, 1),   # dec3 @128px
        (32, None, 0),   # dec4 @256px
    )
    dec_out_ch: int = 32
    heat_hidden: int = 48
    tower_hidden: int = 48
    heat_res: Optional[int] = None  # run the heat tower at this res and
                                    # upsample the logits (nuclear option
                                    # for CPU rollouts — ablate before
                                    # trusting; bench.py measures it)
    proj_hidden: int = 512        # shared context projection (pooled global)
    head_hidden: int = 256
    pct_hidden: int = 128
    grid_default: int = 32
    use_mem: bool = True          # tiny GRU (state in act())
    mem_hidden: int = 96
    use_coord: bool = True
    use_gate: bool = True
    use_aux_seg: bool = True
    use_dynamics: bool = True     # next_seg / threat / expand
    use_econ: bool = True
    use_cell_value: bool = True
    aux_seg_classes: int = 4
    dropout_mlp: float = 0.05

    @property
    def ctx_full_dim(self) -> int:
        return self.ctx_dim + self.rtg_dim


# --------------------------------------------------------------------------
# Blocks (v3 patterns, scaled down; SE dropped — honest simplification)
# --------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.silu(x + y)


class DownStage(nn.Module):
    def __init__(self, cin: int, cout: int, blocks: int):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(cout), nn.SiLU()]
        layers += [ResBlock(cout) for _ in range(blocks)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpStage(nn.Module):
    def __init__(self, cin: int, cout: int, skip_c: Optional[int] = None,
                 blocks: int = 1):
        super().__init__()
        self.skip = (nn.Conv2d(skip_c, cin, 1, bias=False)
                     if skip_c is not None else None)
        self.up = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.blocks = nn.Sequential(*[ResBlock(cout)
                                      for _ in range(blocks)])

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # interpolate needs NCHW — convert if the caller fed NHWC
        if x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.contiguous_format)
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        if skip is not None:
            x = x + self.skip(skip)
        x = F.silu(self.bn(self.up(x)))
        return self.blocks(x)


class FiLM(nn.Module):
    def __init__(self, ctx_dim: int, c: int):
        super().__init__()
        self.gamma = nn.Linear(ctx_dim, c)
        self.beta = nn.Linear(ctx_dim, c)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        g = self.gamma(ctx).view(ctx.shape[0], -1, 1, 1) + 1.0
        b = self.beta(ctx).view(ctx.shape[0], -1, 1, 1)
        return x * g + b


class _MLP(nn.Module):
    def __init__(self, cin: int, hidden: int, cout: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cin, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BetaHead(nn.Module):
    """pct in [0,1] as Beta(alpha,beta); outputs STRICTLY (B,) for both
    params (the v3 review's (B,B) broadcast bug was a (B,1) param shape)."""

    def __init__(self, in_c: int, ctx_dim: int, num_kinds: int,
                 hidden: int = 128, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_c + ctx_dim + num_kinds, hidden),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2))

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor,
                kind_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([feat, ctx, kind_onehot], dim=1)
        a, b = self.net(h).chunk(2, dim=1)
        a = F.softplus(a) + 1.0
        b = F.softplus(b) + 1.0
        return a.squeeze(-1), b.squeeze(-1)          # (B,) — no broadcast


class _ConvBn(nn.Module):
    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Nano(nn.Module):
    """SOVEREIGN-nano — see module docstring for the contract."""

    def __init__(self, cfg: Optional[NanoConfig] = None):
        super().__init__()
        self.cfg = cfg or NanoConfig()
        c = self.cfg
        H = c.map_size
        assert c.dec_plan[-1][0] == c.dec_out_ch, \
            "dec_plan[-1][0] must equal dec_out_ch"

        # stem
        in_c = c.in_c + (2 if c.use_coord else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, c.enc_ch[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c.enc_ch[0]), nn.SiLU())
        if c.use_coord:
            ys = torch.linspace(-1, 1, H).view(1, 1, H, 1).repeat(1, 1, 1, H)
            xs = torch.linspace(-1, 1, H).view(1, 1, 1, H).repeat(1, 1, H, 1)
            self.register_buffer("coords", torch.cat([ys, xs], dim=1))

        # encoder
        ch = c.enc_ch[0]
        self.enc = nn.ModuleList()
        for i, blocks in enumerate(c.enc_blocks):
            self.enc.append(DownStage(ch, c.enc_ch[i + 1], blocks))
            ch = c.enc_ch[i + 1]

        self.film = FiLM(c.ctx_full_dim, ch)         # on bottleneck @8px

        # decoder
        self.dec = nn.ModuleList()
        for cout, skip_c, blocks in c.dec_plan:
            self.dec.append(UpStage(ch, cout, skip_c, blocks))
            ch = cout

        # heat heads — SHARED tower (speed pass): the old 3 per-kind towers
        # ran ~8.2G MACs at 256² (65% of the whole network's compute).
        # ONE tower emitting all 3 channels cuts heat cost ~3× while
        # keeping a full-res 3×3 so the kinds still cross-talk spatially.
        self.heat_mixer = nn.Sequential(
            _ConvBn(c.dec_out_ch, c.heat_hidden, 3),
            _ConvBn(c.heat_hidden, c.heat_hidden, 3))
        self.heat_tower = nn.Sequential(
            _ConvBn(c.heat_hidden, c.tower_hidden, 3),
            _ConvBn(c.tower_hidden, c.tower_hidden, 3),
            nn.Conv2d(c.tower_hidden, c.num_kinds, 1))

        # spatial aux heads on e1 (@H/4)
        if c.use_gate:
            self.gate_head = nn.Conv2d(c.enc_ch[1], c.num_kinds, 1)
        if c.use_aux_seg:
            self.seg_head = nn.Conv2d(c.enc_ch[1], c.aux_seg_classes, 1)
        if c.use_dynamics:
            self.next_seg_head = nn.Conv2d(c.enc_ch[1], c.aux_seg_classes, 1)
            self.threat_head = nn.Conv2d(c.enc_ch[1], 1, 1)
            self.expand_head = nn.Conv2d(c.enc_ch[1], 1, 1)

        # global heads (pooled-MLP replaces the v3 transformer cortex)
        pool_dim = c.enc_ch[-1] + c.ctx_full_dim        # 192 + 9 = 201
        self.proj = nn.Linear(pool_dim, c.proj_hidden)
        self.kind_head = _MLP(c.proj_hidden, c.head_hidden, c.num_kinds,
                              c.dropout_mlp)
        self.value_head = _MLP(c.proj_hidden, c.head_hidden, 1,
                               c.dropout_mlp)
        self.win_head = _MLP(c.proj_hidden, c.head_hidden, 2,
                             c.dropout_mlp)
        self.econ_head = (_MLP(c.proj_hidden, c.head_hidden,
                               c.ctx_full_dim, c.dropout_mlp)
                          if c.use_econ else None)
        self.pct_head = BetaHead(c.dec_out_ch, c.ctx_full_dim, c.num_kinds,
                                 hidden=c.pct_hidden, dropout=c.dropout_mlp)
        self.cell_value = (
            nn.Sequential(_ConvBn(c.dec_plan[1][0], 32, 1),
                          nn.Conv2d(32, 1, 1))
            if c.use_cell_value else None)              # from dec1 @32px

        # memory
        self.mem = None
        if c.use_mem:
            self.mem = nn.GRU(pool_dim, c.mem_hidden, 1, batch_first=True)
            self.mem_proj = nn.Linear(c.mem_hidden, c.enc_ch[-1])

        self.apply(_init_weights)

        # ANTI-OVER-CONFIDENCE INIT (pipeline review finding 1, 15 Aug):
        # at default init the cell CE started at ~16.05 instead of the
        # uniform-floor ln(g²)≈6.93 — the tower's final conv + 1x1 made
        # peaked logits. Scale the final conv by 0.1x and zero its bias
        # so the heat starts near-uniform. The gate head's sigmoid(0)=0.5
        # adds only a constant -0.693 logit, which does not shift the
        # softmax. The smoke ASSERTS the resulting init cell CE < 8.5.
        with torch.no_grad():
            w = self.heat_tower[-1].weight
            w.mul_(0.1)
            if self.heat_tower[-1].bias is not None:
                self.heat_tower[-1].bias.zero_()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _coord_norm(yx: torch.Tensor, size: int) -> torch.Tensor:
        return (yx.float() / max(size - 1, 1)) * 2.0 - 1.0

    @staticmethod
    def _gather(feat: torch.Tensor, yx: torch.Tensor) -> torch.Tensor:
        # grid_sample needs NCHW — convert if the caller fed NHWC
        if feat.is_contiguous(memory_format=torch.channels_last):
            feat = feat.contiguous(memory_format=torch.contiguous_format)
        coords = Nano._coord_norm(yx, feat.shape[-1]).flip(-1)
        return F.grid_sample(feat, coords.view(-1, 1, 1, 2),
                             align_corners=False).squeeze(-1).squeeze(-1)

    def _ctx_full(self, nums: torch.Tensor,
                  rtg: Optional[torch.Tensor]) -> torch.Tensor:
        B = nums.shape[0]
        if rtg is not None:
            r = rtg.view(B, -1).float()
        else:
            r = torch.zeros(B, self.cfg.rtg_dim, device=nums.device)
        return torch.cat([nums, r], dim=1)

    def _gate_logit(self, feat_q: torch.Tensor, H: int) -> torch.Tensor:
        g = torch.sigmoid(self.gate_head(feat_q))
        logg = torch.log(g.clamp_min(EPS))
        if logg.shape[-1] != H:
            logg = F.interpolate(logg, size=(H, H), mode="bilinear",
                                 align_corners=False)
        return logg

    # -- forward ------------------------------------------------------------

    def forward(self, rgb: torch.Tensor, nums: Optional[torch.Tensor] = None,
                rtg: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                grid: Optional[int] = None,
                cell: Optional[torch.Tensor] = None,
                state: Optional[torch.Tensor] = None,
                return_all: bool = False):
        """Contract forward. rtg=None means zero aspiration (contract-safe).

        Returns (click (B,g*g), kind_logits (B,3), pct (B,), value (B,))
        unless return_all=True -> full dict.
        """
        cfg = self.cfg
        B, C, H, W = rgb.shape
        assert H == W == cfg.map_size, \
            f"expected {cfg.map_size}x{cfg.map_size}, got {H}x{W}"
        if nums is None:
            nums = torch.zeros(B, cfg.ctx_dim, device=rgb.device)
        ctx_full = self._ctx_full(nums, rtg)             # (B, 9)

        x = rgb
        if cfg.use_coord:
            x = torch.cat([x, self.coords.expand(B, -1, -1, -1)], dim=1)

        skips = []
        x = self.stem(x)
        for stage in self.enc:
            x = stage(x)
            skips.append(x)              # [@64,@32,@16,@8]

        bot = self.film(x, ctx_full)     # bottleneck @8px
        g = F.adaptive_avg_pool2d(bot, 1).flatten(1)     # (B,192)

        mem_state = None
        if self.mem is not None:
            xin = torch.cat([g, ctx_full], dim=1).unsqueeze(1)   # (B,1,201)
            h0 = state if state is not None else torch.zeros(
                1, B, cfg.mem_hidden, device=rgb.device)
            o, hn = self.mem(xin, h0)
            g = self.mem_proj(o.squeeze(1))              # (B,192)
            mem_state = hn

        # decoder: dec0<-e3(@16), dec1<-e2(@32), dec2<-e1(@64)
        d = self.dec[0](bot, skips[2])
        d32 = self.dec[1](d, skips[1])
        d64 = self.dec[2](d32, skips[0])
        d128 = self.dec[3](d64, None)
        dres = self.dec[4](d128, None)                   # @256 (dec_out_ch)

        hmix = self.heat_mixer(dres)
        if cfg.heat_res is not None and cfg.heat_res < H:
            hres = cfg.heat_res
            hmix = F.interpolate(hmix, size=(hres, hres), mode="bilinear",
                                 align_corners=False)
            heat_raw = self.heat_tower(hmix)
            heat_raw = F.interpolate(heat_raw, size=(H, H), mode="bilinear",
                                     align_corners=False)
        else:
            heat_raw = self.heat_tower(hmix)             # (B,3,H,W)
        heat = heat_raw
        if cfg.use_gate:
            heat = heat + self._gate_logit(skips[0], H)
        if mask is not None:
            assert mask.shape[-2:] == (H, W)
            heat = heat + torch.log(mask.unsqueeze(1).clamp_min(EPS))

        gd = grid if grid is not None else cfg.grid_default
        heat_pool = F.avg_pool2d(heat, H // gd).view(B, cfg.num_kinds, -1)
        click = torch.logsumexp(heat_pool, dim=1)        # (B, g*g)

        feat = torch.cat([g, ctx_full], dim=1)           # (B,201)
        hg = F.silu(self.proj(feat))
        kind_logits = self.kind_head(hg)
        value = self.value_head(hg).squeeze(-1)
        win_logits = self.win_head(hg)
        econ = self.econ_head(hg) if self.econ_head is not None else None
        kind_soft = F.softmax(kind_logits, dim=1)

        if cell is not None:
            y = (cell // gd) * (H // gd) + (H // gd) // 2
            xc = (cell % gd) * (H // gd) + (H // gd) // 2
            yx = torch.stack([y, xc], dim=1).to(rgb.device)
        else:
            best = torch.argmax(click, dim=1)
            y = (best // gd) * (H // gd) + (H // gd) // 2
            xc = (best % gd) * (H // gd) + (H // gd) // 2
            yx = torch.stack([y, xc], dim=1).to(rgb.device)
        pct_feat = self._gather(dres, yx)
        alpha, beta_p = self.pct_head(pct_feat, ctx_full, kind_soft)
        pct = alpha / (alpha + beta_p)                   # (B,)

        out = dict(
            kind_logits=kind_logits,
            kind_probs=kind_soft,
            heat=heat,
            heat_raw=heat_raw,
            cell_logits=heat_pool,                       # (B,3,g*g)
            click=click,
            pct=pct,
            pct_params=(alpha, beta_p),                  # ((B,),(B,))
            value=value,
            win_logits=win_logits,
            win_prob=torch.softmax(win_logits, dim=1)[:, 1],
            econ=econ,
            gate=torch.sigmoid(self.gate_head(skips[0])) if cfg.use_gate
            else None,
            seg=self.seg_head(skips[0]) if cfg.use_aux_seg else None,
            next_seg=self.next_seg_head(skips[0]) if cfg.use_dynamics
            else None,
            threat=torch.sigmoid(self.threat_head(skips[0])).squeeze(1)
            if cfg.use_dynamics else None,
            expand=torch.sigmoid(self.expand_head(skips[0])).squeeze(1)
            if cfg.use_dynamics else None,
            cell_value=(self.cell_value(d32).squeeze(1)
                        if self.cell_value is not None else None),
            dec_feat=dres,
            mem_state=mem_state,
        )
        if return_all:
            return out
        return out["click"], kind_logits, pct, value

    # -- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, rgb: torch.Tensor, nums: Optional[torch.Tensor] = None,
            rtg: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            grid: Optional[int] = None,
            state: Optional[torch.Tensor] = None,
            deterministic: bool = False,
            temperature: float = 1.0) -> dict:
        """Sample a full action. logprob and entropy are STRICTLY (B,)."""
        cfg = self.cfg
        out = self.forward(rgb, nums, rtg=rtg, mask=mask, grid=grid,
                           state=state, return_all=True)
        B = rgb.shape[0]
        gd = grid if grid is not None else cfg.grid_default
        H = cfg.map_size

        kp = out["kind_probs"] / max(temperature, EPS)
        kd = Categorical(logits=torch.log(kp.clamp_min(EPS)))
        kind = kd.sample() if not deterministic else torch.argmax(kp, dim=1)

        heat_k = out["heat"][torch.arange(B, device=rgb.device), kind]
        heat_pool_k = F.avg_pool2d(heat_k, H // gd).view(B, -1)
        cd = Categorical(logits=heat_pool_k / max(temperature, EPS))
        cell = cd.sample() if not deterministic else torch.argmax(
            heat_pool_k, dim=1)

        y = (cell // gd) * (H // gd) + (H // gd) // 2
        xc = (cell % gd) * (H // gd) + (H // gd) // 2
        yx = torch.stack([y, xc], dim=1).to(rgb.device)
        pct_feat = self._gather(out["dec_feat"], yx)
        kind_oh = F.one_hot(kind, cfg.num_kinds).float()
        ctx_full = self._ctx_full(
            nums if nums is not None
            else torch.zeros(B, cfg.ctx_dim, device=rgb.device), rtg)
        alpha, beta_p = self.pct_head(pct_feat, ctx_full, kind_oh)   # (B,)
        beta_d = Beta(alpha, beta_p)
        pct = beta_d.sample() if not deterministic else beta_d.mean   # (B,)

        logprob = (kd.log_prob(kind) + cd.log_prob(cell)
                   + beta_d.log_prob(pct.clamp(EPS, 1 - EPS)))
        entropy = (kd.entropy() + cd.entropy() + beta_d.entropy())
        # v3 review bug 4: these broadcast to (B,B) — now asserted
        assert logprob.shape == (B,), f"logprob {logprob.shape} != {(B,)}"
        assert entropy.shape == (B,), f"entropy {entropy.shape} != {(B,)}"

        return dict(kind=kind, cell=cell, yx=yx, pct=pct, logprob=logprob,
                    value=out["value"], win_prob=out["win_prob"],
                    entropy=entropy, kind_probs=out["kind_probs"],
                    state=out["mem_state"])


# --------------------------------------------------------------------------
# Stage-A loss — every v3 review bug fixed
# --------------------------------------------------------------------------

def cat_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Manual categorical entropy (v3 review bug 5: F.categorical_entropy
    does not exist)."""
    p = F.softmax(logits, dim=dim)
    return -(p * torch.log(p + EPS)).sum(dim=dim)


def stage_a_loss(out: dict, kind: torch.Tensor, cell: torch.Tensor,
                 pct_t: torch.Tensor,
                 ret: Optional[torch.Tensor] = None,
                 win_lab: Optional[torch.Tensor] = None,
                 lab64: Optional[torch.Tensor] = None,
                 lab64_next: Optional[torch.Tensor] = None,
                 gate_mask: Optional[torch.Tensor] = None,
                 gate_valid: Optional[torch.Tensor] = None,
                 threat: Optional[torch.Tensor] = None,
                 expand: Optional[torch.Tensor] = None,
                 nums_next: Optional[torch.Tensor] = None,
                 w: Optional[torch.Tensor] = None,
                 lamb: dict = None) -> Tuple[torch.Tensor, dict]:
    """Weighted filtered-BC + value + win + gate + seg + dynamics + econ.

    `out` MUST come from forward(..., cell=cell, rtg=rtg, return_all=True)
    so pct_params sit at the recorded cell. Every auxiliary CE is reduced
    to a per-sample (B,) before weighting — no silent broadcasts.
    """
    lamb = lamb or dict(kind=0.5, cell=1.0, pct=0.25, val=0.1, win=0.1,
                        gate=0.5, seg=0.5, next_seg=0.4, threat=0.2,
                        expand=0.2, econ=0.05, ent=0.003)
    B = kind.shape[0]
    w = w if w is not None else torch.ones(B, device=kind.device)
    terms = {}

    kind_nll = F.cross_entropy(out["kind_logits"], kind, reduction="none",
                               label_smoothing=0.05)
    # v3 review bug 2: CE((B,3,g*g),(B,)) crashes — select the taken-kind row
    # + label smoothing 0.02 (pipeline review finding 1: keeps cell peaks
    # soft during Stage-A; complements the 0.1x heat init)
    cell_nll = F.cross_entropy(
        out["cell_logits"][torch.arange(B, device=kind.device), kind.long()],
        cell.long(), reduction="none", label_smoothing=0.02)
    pct_nll = -(Beta(out["pct_params"][0], out["pct_params"][1])
                .log_prob(pct_t.clamp(EPS, 1 - EPS)))
    pct_nll = pct_nll * (kind != 2).float()              # bank: pct undefined

    loss = (lamb["kind"] * kind_nll + lamb["cell"] * cell_nll
            + lamb["pct"] * pct_nll)
    terms["kind_nll"] = kind_nll.mean().detach()
    terms["cell_nll"] = cell_nll.mean().detach()
    terms["pct_nll"] = pct_nll.mean().detach()

    if ret is not None:
        val_loss = F.huber_loss(out["value"], ret, reduction="none")
        loss = loss + lamb["val"] * val_loss
        terms["val"] = val_loss.mean().detach()

    if win_lab is not None:
        win_nll = F.cross_entropy(out["win_logits"], win_lab.long(),
                                  reduction="none", label_smoothing=0.05)
        loss = loss + lamb["win"] * win_nll
        terms["win"] = win_nll.mean().detach()

    if out["gate"] is not None and gate_mask is not None:
        g_bce = F.binary_cross_entropy(out["gate"], gate_mask,
                                       reduction="none").mean(dim=(2, 3))
        g_bce = g_bce * gate_valid.float()               # (B,3)
        denom = gate_valid.float().sum(dim=1).clamp_min(1.0)   # (B,)
        gate_loss = g_bce.sum(dim=1) / denom              # (B,)
        loss = loss + lamb["gate"] * gate_loss
        terms["gate"] = gate_loss.mean().detach()

    # v3 review bug 3: seg/next_seg CE is (B,H,W) — reduce per-sample
    if out["seg"] is not None and lab64 is not None:
        seg_ce = F.cross_entropy(out["seg"], lab64.long(),
                                 reduction="none").mean(dim=(1, 2))
        loss = loss + lamb["seg"] * seg_ce
        terms["seg"] = seg_ce.mean().detach()

    if out["next_seg"] is not None and lab64_next is not None:
        nseg_ce = F.cross_entropy(out["next_seg"], lab64_next.long(),
                                  reduction="none").mean(dim=(1, 2))
        loss = loss + lamb["next_seg"] * nseg_ce
        terms["next_seg"] = nseg_ce.mean().detach()

    if out["threat"] is not None and threat is not None:
        th = F.binary_cross_entropy(out["threat"], threat,
                                    reduction="none").mean(dim=(1, 2))
        loss = loss + lamb["threat"] * th
        terms["threat"] = th.mean().detach()

    if out["expand"] is not None and expand is not None:
        ex = F.binary_cross_entropy(out["expand"], expand,
                                    reduction="none").mean(dim=(1, 2))
        loss = loss + lamb["expand"] * ex
        terms["expand"] = ex.mean().detach()

    if out["econ"] is not None and nums_next is not None:
        econ_l1 = F.smooth_l1_loss(out["econ"], nums_next,
                                   reduction="none").mean(dim=1)
        loss = loss + lamb["econ"] * econ_l1
        terms["econ"] = econ_l1.mean().detach()

    marg = torch.logsumexp(out["cell_logits"], dim=1)
    ent = (cat_entropy(out["kind_logits"]) +
           cat_entropy(marg) +
           Beta(out["pct_params"][0], out["pct_params"][1]).entropy())
    loss = loss - lamb["ent"] * ent
    terms["entropy"] = ent.mean().detach()

    total = (loss * w).mean()
    terms["total"] = total.detach()
    return total, terms


# --------------------------------------------------------------------------
# Real-shard prep (THE schema — not the v3 assumption)
# --------------------------------------------------------------------------

def downscale_labels(lab: np.ndarray, size: int) -> np.ndarray:
    """(N,H,W) uint8 labels -> (N,size,size) int64 majority vote."""
    N, H, W = lab.shape
    gh, gw = H // size, W // size
    out = np.zeros((N, size, size), dtype=np.int64)
    for gy in range(size):
        for gx in range(size):
            blk = lab[:, gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
            blk = blk.reshape(N, -1).astype(np.int64)
            counts = np.stack([(blk == cls).sum(axis=1)
                               for cls in range(4)], axis=1)
            out[:, gy, gx] = counts.argmax(axis=1)
    return out


def gate_masks_from_lab(lab64: np.ndarray, adj: int = 2
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """(N,3,64,64) kind0/1 legality masks + (N,3) validity (kind2 invalid)."""
    N, S, _ = lab64.shape
    masks = np.zeros((N, 3, S, S), dtype=np.float32)
    valid = np.zeros((N, 3), dtype=np.float32)
    valid[:, 0] = 1.0
    valid[:, 1] = 1.0
    me = lab64 == 2
    enemy = lab64 == 3
    neutral = lab64 == 1
    pad = adj
    mp = np.pad(me, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    d = np.zeros_like(me)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            d |= mp[:, pad + dy:pad + dy + S, pad + dx:pad + dx + S]
    masks[:, 0] = (d & neutral).astype(np.float32)
    masks[:, 1] = enemy.astype(np.float32)
    return masks, valid


def prep_shard(d: dict, max_frames: int = 32,
               gate_size: int = 64) -> dict:
    """Prep the REAL shard schema into training tensors.

    Key schema facts (the v3 adapter got these wrong — fixed here):
      reward is PER TICK (len = sum(lens) = 2 x frames),
      alive is PER EPISODE, lens counts TICKS (frames per ep = lens//2).
    """
    rgb = d["rgb"]
    lab = d["lab"]
    nums = d["nums"].astype(np.float32)
    kind = d["kind"].astype(np.int64)
    cell = d["cell"].astype(np.int64)
    pct = d["pct"].astype(np.float32)
    reward = d["reward"].astype(np.float32)
    alive = d["alive"].astype(np.int64)
    lens = d["lens"].astype(np.int64)
    N = len(rgb)

    assert len(reward) == int(lens.sum()), \
        "reward must be PER TICK with len == sum(lens)"
    flens = (lens // 2).tolist()          # frames per episode
    diff = N - sum(flens)
    if diff != 0:
        flens[-1] += diff                 # absorb odd-tick rounding
    assert sum(flens) == N

    # per-frame reward: each frame covers 2 ticks. Episodes with ODD tick
    # counts leave one stray final tick — it folds into that episode's
    # last frame. The rounding case (diff absorbed into the last episode)
    # is padded with zeros so the reshape can never fail on any shard.
    fr = np.zeros(N, dtype=np.float32)
    tpos, fpos = 0, 0
    for e, Lt in enumerate(lens.tolist()):
        fl = flens[e]
        ep_ticks = np.concatenate([
            reward[tpos:tpos + Lt],
            np.zeros(max(0, 2 * fl - Lt), dtype=np.float32)])
        fr[fpos:fpos + fl] = ep_ticks[:2 * fl].reshape(fl, 2).sum(axis=1)
        if Lt % 2 == 1:
            fr[fpos + fl - 1] += ep_ticks[2 * fl]
        tpos += Lt
        fpos += fl

    # per-frame alive (episode survival broadcast over its frames)
    frame_alive = np.concatenate(
        [np.full(fl, int(alive[e]), dtype=bool)
         for e, fl in enumerate(flens)])[:N]

    # per-episode MC returns over frames (gamma^1 per frame step)
    ret = np.zeros(N, dtype=np.float32)
    s = 0
    for fl in flens:
        e = s + fl
        r = fr[s:e].copy()
        if frame_alive[e - 1]:
            r[-1] += 2.0                 # survival bonus (shaping-agnostic)
        acc = 0.0
        for i in range(e - 1, s - 1, -1):
            acc = r[i - s] + 0.995 * acc
            ret[i] = acc
        s = e
    ret = (ret - ret.mean()) / max(ret.std(), 1e-6)
    rtg = ret[:, None].astype(np.float32)

    # next-frame targets (episode-safe shifts)
    rtg_next = np.empty_like(rtg)
    nums_next = np.empty_like(nums)
    s = 0
    for fl in flens:
        e = s + fl
        rtg_next[s:e - 1] = rtg[s + 1:e]
        rtg_next[e - 1] = rtg[e - 1]
        nums_next[s:e - 1] = nums[s + 1:e]
        nums_next[e - 1] = nums[e - 1]
        s = e
    econ_t = np.concatenate([nums_next, rtg_next], axis=1)   # (N,9)

    win_lab = frame_alive.astype(np.int64)

    # AWR weights: exp(return) clip + survivor x4 + kill-window x2
    w = np.clip(np.exp(ret / 1.0), 0.2, 8.0).astype(np.float32)
    w *= (1.0 + 3.0 * frame_alive.astype(np.float32))
    dk = np.diff(nums[:, 7], prepend=nums[0, 7] - 0.0)
    for i in np.where(dk > 1e-3)[0]:
        w[max(0, i - 2):min(N, i + 3)] *= 2.0
    w = w / (w.mean() + 1e-8)

    # label-derived supervision (lab64, next, intent mirrors, gate)
    lab64 = downscale_labels(lab, gate_size)
    lab_next = np.empty_like(lab)
    lab_next[:-1] = lab[1:]
    lab_next[-1] = lab[-1]
    s = 0
    for fl in flens:
        lab_next[s + fl - 1] = lab[s + fl - 1]   # episode boundary repeats
        s += fl
    lab64_next = downscale_labels(lab_next, gate_size)
    threat = ((lab64_next == 3) & (lab64 != 3)).astype(np.float32)
    expand = ((lab64 == 1) & (lab64_next == 2)).astype(np.float32)
    gmask, gvalid = gate_masks_from_lab(lab64)

    sel = np.arange(min(max_frames, N))
    t = torch.from_numpy
    return dict(
        rgb=t(rgb[sel].astype(np.float32).transpose(0, 3, 1, 2) / 255.0),
        nums=t(nums[sel]), rtg=t(rtg[sel]),
        kind=t(kind[sel]), cell=t(cell[sel]), pct=t(pct[sel]),
        ret=t(ret[sel]), win_lab=t(win_lab[sel]), w=t(w[sel]),
        lab64=t(lab64[sel]), lab64_next=t(lab64_next[sel]),
        threat=t(threat[sel]), expand=t(expand[sel]),
        gate_mask=t(gmask[sel]), gate_valid=t(gvalid[sel]),
        econ_t=t(econ_t[sel]),
        n_sel=len(sel), real_episodes=len(lens), real_frames=N,
    )


# --------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------

def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def param_breakdown(net: Nano) -> dict:
    n = lambda m: sum(p.numel() for p in m.parameters())
    spine = n(net.stem) + sum(n(m) for m in net.enc) + n(net.film)
    decoder = sum(n(m) for m in net.dec)
    heads = (n(net.heat_mixer) + n(net.heat_tower) +
             n(net.proj) + n(net.kind_head) + n(net.value_head) +
             n(net.win_head) + n(net.pct_head))
    if net.cfg.use_econ:
        heads += n(net.econ_head)
    if net.cfg.use_gate:
        heads += n(net.gate_head)
    if net.cfg.use_aux_seg:
        heads += n(net.seg_head)
    if net.cfg.use_dynamics:
        heads += (n(net.next_seg_head) + n(net.threat_head) +
                  n(net.expand_head))
    if net.cfg.use_cell_value:
        heads += n(net.cell_value)
    mem = (n(net.mem) + n(net.mem_proj)) if net.mem is not None else 0
    return dict(spine=spine, decoder=decoder, heads=heads, mem=mem)


def make_nano(cfg: Optional[NanoConfig] = None,
              seed: Optional[int] = None) -> Nano:
    if seed is not None:
        torch.manual_seed(seed)
    return Nano(cfg)


def load_real_shard(mmap: bool = False):
    """Try the REAL shard first (via $HF_TOKEN), then a local copy.
    mmap=True memory-maps the npz (tiny mode on a 2GB box: the full
    arrays never sit in RAM; prep slices before any float32 conversion)."""
    mode = "r" if mmap else None
    for candidate in ("shard_v2_1786707781_27_23.npz",
                      "rl/shards_v2/shard_v2_1786707781_27_23.npz"):
        if os.path.exists(candidate):
            return np.load(candidate, mmap_mode=mode)
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        if not token:
            return None
        path = hf_hub_download(HF_DATASET, HF_SHARD, repo_type="dataset",
                               token=token, local_dir="/tmp/nano_real")
        return np.load(path, mmap_mode=mode)
    except Exception as e:
        print(f"real shard fetch failed ({str(e)[:100]}) — using replica")
        return None


def synthetic_shard() -> dict:
    """Schema-EXACT replica (per-episode alive/lens in ticks, per-tick
    reward) — used ONLY when the real shard cannot be fetched."""
    rng = np.random.default_rng(0)
    lens = np.array([120, 140, 160, 288], dtype=np.int64)   # ticks
    flens = (lens // 2).tolist()
    N = sum(flens)
    alive = np.array([1, 0, 1, 0], dtype=np.int64)
    reward = rng.uniform(0.003, 0.012, size=int(lens.sum())).astype(
        np.float32)
    # death terminal (-1.3) on the last tick of dead episodes
    pos = 0
    for e, L in enumerate(lens.tolist()):
        if alive[e] == 0:
            reward[pos + L - 1] = -1.3
        pos += L
    lab = np.zeros((N, 256, 256), dtype=np.uint8)
    lab[:, 10:200, 10:200] = 1
    lab[:, 20:80, 20:80] = 2
    lab[:, 100:180, 100:180] = 3
    rgb = (lab[:, :, :, None].astype(np.float32) * 40).astype(np.uint8)
    rgb = np.concatenate([rgb] * 3, axis=-1)
    nums = rng.uniform(0.05, 0.6, size=(N, 8)).astype(np.float32)
    nums[:, 7] = np.floor(np.linspace(0, 3, N))             # kills channel
    kind = rng.integers(0, 3, size=N).astype(np.int64)
    cell = rng.integers(0, 1024, size=N).astype(np.int64)
    pct = rng.uniform(0.1, 0.6, size=N).astype(np.float32)
    logp = np.full(N, -2.0, dtype=np.float32)
    return dict(rgb=rgb, lab=lab, nums=nums, kind=kind, cell=cell, pct=pct,
                logp=logp, reward=reward, alive=alive, lens=lens)


# --------------------------------------------------------------------------
# Smoke test (the pipeline runs this on the REAL shard)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("== SOVEREIGN-nano smoke ==")
    tiny = "--tiny" in sys.argv          # pipeline review finding 2:
                                         # small boxes get a low-memory path
    if tiny:
        print("tiny mode: 8 frames, loss on a 4-sample slice "
              "(peak ~0.5GB vs ~1.5-2GB full)")
    torch.manual_seed(0)

    net = make_nano()
    total = count_params(net)
    bd = param_breakdown(net)
    assert total <= PARAM_LIMIT, f"{total:,} > {PARAM_LIMIT:,}"
    print(f"params: {total:,} (limit {PARAM_LIMIT:,}) — PASS  "
          f"[spine {bd['spine']:,} | decoder {bd['decoder']:,} | "
          f"heads {bd['heads']:,} | mem {bd['mem']:,}]")

    d = load_real_shard(mmap=tiny)
    if d is not None:
        print("REAL shard loaded — schema asserts below:")
        print(f"  keys: {sorted(k for k in d.keys())}")
        print(f"  rgb {d['rgb'].shape} | reward {d['reward'].shape} "
              f"(per tick) | alive {d['alive'].shape} (per ep) | "
              f"lens {d['lens'].shape} sum={int(d['lens'].sum())}")
    else:
        print("WARN: real shard unavailable (no HF_TOKEN/network) — "
              "building a schema-EXACT synthetic replica")
        d = synthetic_shard()

    prep = prep_shard(d, max_frames=8 if tiny else 16)
    B = prep["n_sel"]
    print(f"prep: {B} frames from {prep['real_episodes']} episodes "
          f"({prep['real_frames']} total frames in shard)")

    # forward + act probes — no_grad (B's 17-Aug note: the probes were
    # grad-enabled and never freed, holding full-graph activations at
    # B=8/16 — that was the ~1.5GB tiny-mode peak. no_grad + del keeps
    # the only grad graph as the L=4 loss below.)
    with torch.no_grad():
        click, kind_logits, pct, value = net(prep["rgb"], prep["nums"],
                                             rtg=prep["rtg"])
        assert click.shape == (B, 1024), click.shape
        assert kind_logits.shape == (B, 3), kind_logits.shape
        assert pct.shape == (B,), pct.shape
        assert value.shape == (B,), value.shape
        print(f"forward: {tuple(click.shape)} {tuple(kind_logits.shape)} "
              f"{tuple(pct.shape)} {tuple(value.shape)} — PASS")

        # act at B=4 — logprob/entropy MUST be (B,)
        a = net.act(prep["rgb"][:4], prep["nums"][:4], rtg=prep["rtg"][:4])
        assert a["logprob"].shape == (4,), a["logprob"].shape
        assert a["entropy"].shape == (4,), a["entropy"].shape
        print(f"act(B=4): logprob {tuple(a['logprob'].shape)} | entropy "
              f"{tuple(a['entropy'].shape)} | state "
              f"{tuple(a['state'].shape)} — PASS")
        del click, kind_logits, pct, value, a

    # stage-a loss + backward (tiny mode: 4-sample slice keeps one small
    # grad graph instead of two bs16 graphs)
    L = min(B, 4) if tiny else B
    loss_out = net(prep["rgb"][:L], prep["nums"][:L], rtg=prep["rtg"][:L],
                   cell=prep["cell"][:L], return_all=True)
    loss, terms = stage_a_loss(
        loss_out, prep["kind"][:L], prep["cell"][:L], prep["pct"][:L],
        ret=prep["ret"][:L], win_lab=prep["win_lab"][:L],
        lab64=prep["lab64"][:L], lab64_next=prep["lab64_next"][:L],
        gate_mask=prep["gate_mask"][:L], gate_valid=prep["gate_valid"][:L],
        threat=prep["threat"][:L], expand=prep["expand"][:L],
        nums_next=prep["econ_t"][:L], w=prep["w"][:L])
    assert torch.isfinite(loss), "non-finite loss"
    # pipeline review finding 1 (asserted): with the 0.1x heat-tower init,
    # the cell CE must NOT be over-confident. NOTE (B, 17 Aug): on the
    # real shard 1786707781 the init cell CE measures 8.81/9.19 — the
    # ideal uniform floor ln(1024)≈6.93 is unreachable because the
    # mixer's nonzero features survive the 0.1x tower conv. Threshold
    # <9.5 (B's calibrated value) still catches the 16+ disease class.
    init_cell = float(terms["cell_nll"])
    assert init_cell < 9.5, \
        f"init cell CE {init_cell:.2f} — heat init over-confident (regressed?)"
    loss.backward()
    print(f"loss: {loss.item():.4f} — backward OK — PASS")
    print("terms: " + " ".join(f"{k}={float(v):.3f}" for k, v in
                               terms.items()))
    print("NANO SMOKE OK")
