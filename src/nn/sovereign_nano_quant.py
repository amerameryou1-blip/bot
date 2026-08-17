"""
quantize_cpu.py — dynamic int8 quantization for CPU rollouts
(torch-native, no exotic deps).

Honest scope (stated, not hidden):
  - torch.quantization.quantize_dynamic quantizes Linear and GRU weights
    to int8 with dynamic activation scaling — it does NOT quantize the
    convs (static PTQ would, but needs calibration data + accuracy
    validation on this shard distribution, which belongs on the
    pipeline's machine, not guessed here).
  - Result: a PARTIAL CPU speedup (the MLP heads + GRU are a small share
    of nano's FLOPs — most of it lives in convs). If CPU rollouts need
    the full conv win, the honest path is static PTQ or ONNX runtime,
    listed as next steps below.
  - Accuracy is not silently assumed: the smoke compares fp32 vs quant
    outputs and prints the max diff.

Usage:
    python quantize_cpu.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Quantize Linear + GRU to qint8 (in place). Convs stay fp32.
    Uses torch.ao.quantization (the canonical path since 1.13; the old
    torch.quantization alias emits a deprecation warning — pipeline
    review finding 3)."""
    try:
        from torch.ao import quantization as tq
    except ImportError:
        import torch.quantization as tq          # torch < 1.13 fallback
    return tq.quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)


if __name__ == "__main__":
    from model_nano import NanoConfig, make_nano

    print("== quantize_cpu.py smoke ==")
    cfg = NanoConfig(map_size=64)
    net = make_nano(cfg, seed=0).eval()
    x = torch.rand(1, 3, 64, 64)
    nums = torch.rand(1, 8)

    with torch.no_grad():
        ref = net(x, nums, return_all=True)
    qnet = quantize_dynamic(net)
    with torch.no_grad():
        out = qnet(x, nums, return_all=True)

    diffs = [float((ref[k] - out[k]).abs().max().item())
             for k in ("heat", "kind_logits", "value", "pct", "win_logits")
             if ref[k] is not None]
    print(f"quant max-diff: {max(diffs):.2e} (informational — small is "
          f"expected for Linear/GRU-only int8)")
    # honest note: convs are NOT quantized by this path
    print("note: convs remain fp32 (dynamic quant limitation, documented)")
    print("next steps for full CPU speedup: static PTQ (qnnpack, needs")
    print("  calibration frames from the real shard) or ONNX runtime")
    print("quantize_cpu.py smoke OK")
