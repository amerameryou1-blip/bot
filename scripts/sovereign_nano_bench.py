"""
bench.py — measurable speed claims for SOVEREIGN-nano.

Every optimization in opt.py/model_nano gets a number here. The pipeline
can run this on CPU or 2xT4 and compare against nano_notes.md — if a
claim is off, this file is where it shows.

Usage:
    python bench.py                          # CPU, default config
    python bench.py --device cuda            # T4
    python bench.py --heat-res 64            # nuclear heat option
    python bench.py --fp16 --device cuda     # autocast (cuda only)
    python bench.py --compile                # torch.compile
    python bench.py --channels-last          # NHWC input path
    python bench.py --batch 4 --iters 20

Runs forward + act and prints ms/frame and frames/sec for each config.
All runs are at map_size=256 (the real contract size) by default; use
--small 64 for a fast sanity path.
"""

from __future__ import annotations

import argparse
import time

import torch

from model_nano import NanoConfig, make_nano


def timeit(fn, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_one(cfg, device, args, label: str) -> None:
    net = make_nano(cfg, seed=0).eval().to(device)
    try:
        from opt import fold_bn, to_channels_last
        if args.fold:
            fold_bn(net)
            label += "+fold"
        if args.compile:
            from opt import compile_if_available
            net = compile_if_available(net)
            label += "+compile"
    except ImportError:
        pass

    H = cfg.map_size
    rgb = torch.rand(args.batch, 3, H, H, device=device)
    nums = torch.rand(args.batch, 8, device=device)

    def prep(x):
        if args.channels_last and x.dim() == 4:
            from opt import to_channels_last
            return to_channels_last(x)
        return x

    ctx = None
    if args.fp16 and device.type == "cuda":
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)

    def fwd():
        if ctx is not None:
            with ctx:
                net(prep(rgb), nums)
        else:
            net(prep(rgb), nums)

    def act():
        if ctx is not None:
            with ctx:
                net.act(prep(rgb), nums)
        else:
            net.act(prep(rgb), nums)

    try:
        ms_fwd = timeit(fwd, args.iters, args.warmup, device)
        ms_act = timeit(act, args.iters, args.warmup, device)
        fps = 1000.0 / max(ms_act, 1e-6)
        print(f"{label:38s} fwd {ms_fwd:7.2f} ms | act {ms_act:7.2f} ms "
              f"| {fps:7.1f} acts/s")
    except Exception as e:
        print(f"{label:38s} FAILED: {str(e)[:120]} "
              f"(config unsupported — stated, not hidden)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--small", type=int, default=0,
                    help="use this map size instead of 256 (e.g. 64)")
    ap.add_argument("--heat-res", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--fold", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable — falling back to CPU")
        device = torch.device("cpu")
    print(f"== SOVEREIGN-nano bench | device={device} batch={args.batch} "
          f"iters={args.iters} ==")

    size = args.small or 256
    base = NanoConfig(map_size=size)
    run_one(base, device, args, f"baseline {size}px")

    if args.heat_res:
        hr = NanoConfig(map_size=size, heat_res=args.heat_res)
        run_one(hr, device, args, f"heat_res={args.heat_res}")
    if args.fold:
        run_one(base, device, args, "baseline")
    if args.channels_last:
        run_one(base, device, args, "baseline")
    if args.compile:
        run_one(base, device, args, "baseline")
    if args.fp16:
        run_one(base, device, args, "baseline")
    print("done")


if __name__ == "__main__":
    main()
