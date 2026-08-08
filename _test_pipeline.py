#!/usr/bin/env python3
"""Quick CPU pipeline sanity: collect(small) -> vision(1) -> real(3) — verify the
FIXED real stage learns water/me/enemy from a FRESH sim checkpoint."""
import os, sys, time
os.environ['FORCE_CPU'] = '1'
os.environ['COLLECT_SEEDS'] = '6'
os.environ['WORKERS'] = '2'
os.environ['SIM_BOTS'] = '8'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import torch
from train_nn import DEVICE, stage_collect, stage_vision, stage_real, WEIGHTS, REAL_NPZ
from nn.model import TerritoryNet

net = TerritoryNet()
print("device:", DEVICE)
t0 = time.time()
stage_collect(seeds=6, workers=2)   # writes weights/nn/dataset.npz
print(f"collect done ({(time.time()-t0)/60:.1f} min)")
net = stage_vision(net, epochs=1, bs=64)
print(f"vision done ({(time.time()-t0)/60:.1f} min)")
try:
    net = stage_real(net, epochs=3, bs=64, lr=5e-5)
    print(f"PIPELINE CPU TEST: REAL STAGE PASSED ({(time.time()-t0)/60:.1f} min)")
except RuntimeError as e:
    print(f"PIPELINE CPU TEST: GATE FAILED LOUDLY -> {e}")
    sys.exit(1)
