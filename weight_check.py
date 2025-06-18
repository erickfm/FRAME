#!/usr/bin/env python3
"""
weight_check.py  –  Audit a FRAME checkpoint for NaN / Inf weights.

Usage:
    python weight_check.py ./checkpoints/epoch_09.pt
"""

import argparse
import sys
import torch

# ── import your model code ──────────────────────────────────────────────────
from model import FramePredictor, ModelConfig   # adjust path if needed

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Check checkpoint for NaN/Inf weights")
parser.add_argument("ckpt_path", type=str, help="Path to .pt checkpoint file")
args = parser.parse_args()

# ── load ────────────────────────────────────────────────────────────────────
try:
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
except FileNotFoundError:
    print(f"❌  File not found: {args.ckpt_path}")
    sys.exit(1)

cfg   = ModelConfig(**ckpt["config"])
model = FramePredictor(cfg)
model.load_state_dict(ckpt["model_state_dict"], strict=False)

# ── scan weights ────────────────────────────────────────────────────────────
nan_total = 0
inf_total = 0
bad_layers = []

for name, p in model.named_parameters():
    n_nan = torch.isnan(p).sum().item()
    n_inf = torch.isinf(p).sum().item()
    if n_nan or n_inf:
        bad_layers.append((name, n_nan, n_inf))
        nan_total += n_nan
        inf_total += n_inf

# ── report ──────────────────────────────────────────────────────────────────
if nan_total == 0 and inf_total == 0:
    print(f"✅  {args.ckpt_path} is clean (no NaN / Inf weights).")
else:
    print(f"⚠️   Problems found in {args.ckpt_path}:")
    print(f"    NaN weights: {nan_total:,}")
    print(f"    Inf weights: {inf_total:,}")
    print("\n  Offending layers (up to 10 shown):")
    for name, n_nan, n_inf in bad_layers[:10]:
        print(f"    {name:<40}  NaN={n_nan:<6}  Inf={n_inf}")

    if len(bad_layers) > 10:
        print(f"    … and {len(bad_layers) - 10} more.")
