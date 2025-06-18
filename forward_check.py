#!/usr/bin/env python3
# forward_check.py  —  Check for NaNs on a cold model → first batch

import torch
from torch.utils.data import DataLoader

from dataset import MeleeFrameDatasetWithDelay
from train import collate_fn, get_model

def main():
    # 1) Build dataset + dataloader
    ds = MeleeFrameDatasetWithDelay(
        parquet_dir="./data",
        sequence_length=30,
        reaction_delay=1,
    )
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # 2) Grab one batch
    state, target = next(iter(dl))

    # 3) Load the model (on the same device it was trained with)
    model, cfg = get_model()
    model.eval()

    # 4) Move inputs to model’s device
    device = next(model.parameters()).device
    for k, v in state.items():
        state[k] = v.to(device, non_blocking=True)
    for k, v in target.items():
        target[k] = v.to(device, non_blocking=True)

    # 5) Forward and check
    with torch.no_grad():
        preds = model(state)

    bad = False
    for name, t in preds.items():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        if n_nan or n_inf:
            print(f"⚠️  {name}  NaNs={n_nan}  Infs={n_inf}")
            bad = True

    if not bad:
        print("✅  Forward pass produced no NaNs/Infs")

if __name__ == "__main__":
    main()
