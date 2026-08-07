#!/usr/bin/env python3
"""Sanity-check the trained CNN's vision: feed synthetic frames, print the
predicted segmentation map (does it find 'me' where the red block is?)."""
import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import torch
from nn.bot_brain import load_model, NeuralBrain

net = load_model()
if net is None:
    print("no model found"); sys.exit(1)
net.eval()

def synth(me_color=(220,60,60), me_rect=(20,44,20,44), enemy_color=(60,140,240), enemy_rect=(44,60,44,60)):
    img = np.zeros((64,64,3), dtype=np.uint8)
    img[:] = (150,142,120)          # neutral
    img[0:12,:] = (40,90,170)       # water strip top
    y0,y1,x0,x1 = me_rect; img[y0:y1,x0:x1] = me_color
    ey0,ey1,ex0,ex1 = enemy_rect; img[ey0:ey1,ex0:ex1] = enemy_color
    return img

def show(probs):
    cls = int(probs[0].argmax(dim=0))
    # 'me' class = 2, enemy = 3, water = 0, neutral = 1
    chars = {0:'~',1:'.',2:'M',3:'E',4:'U'}
    for r in range(0, 64, 4):
        row = ''.join(chars.get(int(cls[r, c]), '?') for c in range(0, 64, 4))
        print(row)

for name, img in [("me-red + enemy-blue", synth())]:
    x = torch.tensor(img.transpose(2,0,1)[None], dtype=torch.float32)/255.0
    with torch.no_grad():
        seg, *_ = net.forward(x, None, return_all=True)
        probs = torch.softmax(seg, dim=1)
    print(f"--- {name}: predicted segmentation (M=me, E=enemy, ~=water, .=neutral) ---")
    show(probs)
    me_frac = float(probs[0,2].mean()); en_frac = float(probs[0,3].mean())
    print(f"me_frac={me_frac:.3f} enemy_frac={en_frac:.3f}")
    # centroid of 'me' class
    m = probs[0,2]
    ys, xs = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
    tot = m.sum()
    if tot > 0.01:
        cy = (ys.float()*m).sum()/tot; cx = (xs.float()*m).sum()/tot
        print(f"predicted my centroid cell=({cy:.1f},{cx:.1f}) (true block ~ (2..5, 2..5))")
    # action
    from nn.bot_brain import NeuralBrain
    b = NeuralBrain(net)
    act = b.decide(img)
    print("action:", act.kind, act.reason, f"at ({act.x:.0f},{act.y:.0f}) pct={act.pct:.0f}")
