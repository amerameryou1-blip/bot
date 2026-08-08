#!/usr/bin/env python3
"""CPU sanity test of the FIXED real stage (2 epochs) — do NOT run on GPU until this passes."""
import os, sys, time
os.environ['FORCE_CPU'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import torch
from train_nn import stage_real
from nn.model import TerritoryNet

net = TerritoryNet()
net.load_state_dict(torch.load('weights/nn/model.pt', map_location='cpu'))
print("model loaded (v5 model, 85533 params)")
t0 = time.time()
try:
    net = stage_real(net, epochs=4, bs=64, lr=5e-5)
    print(f"REAL STAGE CPU TEST: OK ({(time.time() - t0) / 60:.1f} min)")
except RuntimeError as e:
    print(f"REAL STAGE CPU TEST: GATE FAILED LOUDLY -> {e}")
    sys.exit(1)
