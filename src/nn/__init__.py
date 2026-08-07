"""Neural network package for the territorial.io bot.

TerritoryNet: a small CNN that reads the raw screen image and outputs
  - a per-pixel semantic segmentation (water / neutral / me / enemy / UI)
  - my position on the map (self-localization)
  - an action policy (which cell to click, expand/attack/bank, attack %)
  - a value estimate (for reinforcement learning)

Lightweight: ~100k params, runs on CPU in milliseconds.
"""
