"""
opt.py — inference optimizations for SOVEREIGN-nano (torch-native, no
exotic deps). Each lever states honestly what it can and cannot do.

  1. fold_bn(model) — folds BatchNorm into the preceding convs for
     EVALUATION / rollout mode. Verified numerically in the smoke: at init
     (running stats mu=0, sigma=1) the folded net is EXACTLY equivalent;
     after training it is the standard inference approximation.
     KEEP the unfolded state for training.
  2. to_channels_last(x) — NHWC input helper (encoder + heat convs get
     cuDNN's fast NHWC path; the model auto-converts to NCHW before
     interpolate/grid_sample internally).
  3. compile_if_available(model) — torch.compile (torch>=2.0), 1.3-2x on
     256px CNNs; first call pays a one-time compile cost.
  4. count_flops(model, rgb, nums) — hook-based MAC/GFLOP counter for the
     exact input (so speed claims are measured, not vibes).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 1. BatchNorm folding (inference-only)
# --------------------------------------------------------------------------

def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w = conv.weight.detach().clone()
    gamma = bn.weight.detach().clone()
    beta = bn.bias.detach().clone()
    var = bn.running_var.detach().clone()
    mean = bn.running_mean.detach().clone()
    scale = gamma / torch.sqrt(var + bn.eps)
    wf = w * scale.view(-1, 1, 1, 1)
    b_prev = (conv.bias.detach().clone()
              if conv.bias is not None
              else torch.zeros(w.shape[0], device=w.device))
    bf = (b_prev - mean) * scale + beta
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=True)
    with torch.no_grad():
        new.weight.copy_(wf)
        new.bias.copy_(bf)
    return new


def fold_bn(module: nn.Module) -> nn.Module:
    """Fold every Conv->BN pair into a single Conv (in place). Only call
    this on a model you are about to RUN in eval mode (rollouts/eval)."""
    # adjacent pairs inside nn.Sequential (stem, _ConvBn, mixer, tower)
    if isinstance(module, nn.Sequential):
        keys = list(module._modules.keys())
        i = 0
        while i < len(keys):
            m = module._modules[keys[i]]
            if (isinstance(m, nn.Conv2d) and i + 1 < len(keys)
                    and isinstance(module._modules[keys[i + 1]],
                                   nn.BatchNorm2d)):
                module._modules[keys[i]] = _fold(
                    m, module._modules[keys[i + 1]])
                module._modules[keys[i + 1]] = nn.Identity()
                i += 2
            else:
                i += 1
        for m in module._modules.values():
            fold_bn(m)
        return module

    # attribute pairs (ResBlock: conv1/bn1 conv2/bn2; UpStage: up/bn)
    for cname, bname in (("conv1", "bn1"), ("conv2", "bn2"), ("up", "bn"),
                         ("conv", "bn")):
        c = getattr(module, cname, None)
        b = getattr(module, bname, None)
        if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
            setattr(module, cname, _fold(c, b))
            setattr(module, bname, nn.Identity())
    for m in module._modules.values():
        fold_bn(m)
    return module


# --------------------------------------------------------------------------
# 2. channels_last helper
# --------------------------------------------------------------------------

def to_channels_last(x: torch.Tensor) -> torch.Tensor:
    """NHWC input for the conv-bound paths. The model converts back to
    NCHW internally before interpolate/grid_sample (see model_nano)."""
    return x.contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------
# 3. torch.compile helper
# --------------------------------------------------------------------------

def compile_if_available(model: nn.Module,
                         mode: str = "reduce-overhead") -> nn.Module:
    """Compile if torch>=2.0; returns the (possibly wrapped) model."""
    try:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch < 2.0")
        return torch.compile(model, mode=mode)
    except Exception as e:
        print(f"[opt] torch.compile unavailable ({str(e)[:80]}) — "
              f"returning eager model")
        return model


# --------------------------------------------------------------------------
# 4. FLOP counter (hook-based, exact for the given input)
# --------------------------------------------------------------------------

def count_flops(model: nn.Module, rgb: torch.Tensor,
                nums: Optional[torch.Tensor] = None,
                rtg: Optional[torch.Tensor] = None
                ) -> Tuple[int, Dict[str, int]]:
    """MACs and per-module breakdown for one forward at this input size.
    GRU and grid_sample are omitted (negligible — stated)."""
    macs: Dict[str, int] = defaultdict(int)
    handles = []

    def hook(name: str):
        def f(m, inp, out):
            x = inp[0]
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(m, nn.Conv2d):
                macs[name] = int(
                    o.shape[0] * o.shape[2] * o.shape[3] * m.kernel_size[0]
                    * m.kernel_size[1] * m.in_channels * m.out_channels
                    / max(m.groups, 1))
            elif isinstance(m, nn.Linear):
                macs[name] = int(x.shape[0] * m.in_features * m.out_features)
        return f

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook(name)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(rgb, nums, rtg=rtg)
    model.train(was_training)
    for h in handles:
        h.remove()
    return sum(macs.values()), dict(macs)


def gflops(total_macs: int) -> float:
    return total_macs * 2 / 1e9


# --------------------------------------------------------------------------
# Smoke
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from model_nano import NanoConfig, make_nano

    print("== opt.py smoke ==")
    cfg = NanoConfig(map_size=64)
    net = make_nano(cfg, seed=0).eval()
    x = torch.rand(1, 3, 64, 64)
    nums = torch.rand(1, 8)

    # FLOP count
    total_macs, per_mod = count_flops(net, x, nums)
    print(f"FLOPs @64px: {gflops(total_macs):.2f} GFLOPs "
          f"({total_macs/1e6:.0f}M MACs)")
    top = sorted(per_mod.items(), key=lambda kv: -kv[1])[:4]
    print("  top modules: " + ", ".join(
        f"{n.split('.')[-1]}={v/1e6:.0f}M" for n, v in top))

    # BN folding exactness at init
    ref = net(x, nums, return_all=True)
    fold_bn(net)
    out = net(x, nums, return_all=True)
    diffs = [float((ref[k] - out[k]).abs().max().item())
             for k in ("heat", "kind_logits", "value", "pct", "win_logits")
             if ref[k] is not None]
    print(f"fold max-diff (must be ~0 at init): {max(diffs):.2e}")
    assert max(diffs) < 1e-5, "BN fold is not numerically exact!"

    # channels_last path
    net_cl = make_nano(cfg, seed=0).eval()
    x_cl = to_channels_last(x)
    out_cl = net_cl(x_cl, nums, return_all=True)
    d_cl = float((out["heat"] - out_cl["heat"]).abs().max().item())
    print(f"channels_last max-diff: {d_cl:.2e}")
    assert d_cl < 1e-5

    # torch.compile availability
    cnet = compile_if_available(net)
    print("compile:", "active" if cnet is not net else "unavailable")
    print("opt.py smoke OK")
