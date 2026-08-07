"""Smoke tests for the neural network (shape checks, no training)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nn.model import TerritoryNet, NUM_CLASSES


@pytest.fixture
def net():
    return TerritoryNet(grid=16, context_dim=3)


def test_forward_shapes(net):
    x = torch.randn(4, 3, 64, 64)
    ctx = torch.randn(4, 3)
    seg, local, click, kind, pct, value = net.forward(x, ctx, return_all=True)
    assert seg.shape == (4, NUM_CLASSES, 16, 16)
    assert local.shape == (4, 2)
    assert click.shape == (4, 256)
    assert kind.shape == (4, 3)
    assert pct.shape == (4,)
    assert value.shape == (4,)


def test_forward_without_context(net):
    x = torch.randn(2, 3, 64, 64)
    seg, local, click, kind, pct, value = net.forward(x, None, return_all=True)
    assert seg.shape == (2, NUM_CLASSES, 16, 16)


def test_act_returns_valid_indices(net):
    x = torch.randn(1, 3, 64, 64)
    kind_i, cell, pct, value = net.act(x, torch.randn(1, 3), greedy=True)
    assert kind_i in (0, 1, 2)
    assert 0 <= cell < 256
    assert 0.0 <= pct <= 1.0


def test_segmentation_on_synthetic_frame(net):
    """A synthetic 'red territory on beige land' image should segment 'me'."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:] = (150, 142, 120)  # neutral land
    img[20:40, 20:40] = (220, 60, 60)  # my territory
    x = torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32) / 255.0
    probs = net.seg_probs(x)  # (1,5,16,16)
    # before training the net is random; just verify it runs and each pixel
    # sums to 1 over classes
    per_pixel = probs[0].sum(dim=0)
    assert float(per_pixel.max()) <= 1.0 + 1e-3
    assert float(per_pixel.min()) >= 1.0 - 1e-3
