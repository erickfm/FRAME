#!/usr/bin/env python3
# train.py  —  Debug training with loss + gradient inspection + checkpointing

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
import wandb

from dataset import MeleeFrameDatasetWithDelay
from model import FramePredictor, ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE      = 128
NUM_EPOCHS      = 10
LEARNING_RATE   = 3e-3
NUM_WORKERS     = 8
SEQUENCE_LENGTH = 30
REACTION_DELAY  = 1
DATA_DIR        = "./data"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Collate
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    batch_state, batch_target = {}, {}
    for k in batch[0][0]:
        batch_state[k] = torch.stack([item[0][k] for item in batch], 0)
    for k in batch[0][1]:
        batch_target[k] = torch.stack([item[1][k] for item in batch], 0)
    return batch_state, batch_target

# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss (with per-term safety)
# ─────────────────────────────────────────────────────────────────────────────
def safe_loss(fn, pred, tgt, name):
    out = fn(pred, tgt)
    if not torch.isfinite(out):
        raise RuntimeError(f"❌ {name} produced non-finite value: {out}")
    return out

def compute_loss(preds, targets):
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    main_pred = preds["main_xy"]
    l_pred    = preds["L_val"].squeeze(-1)
    r_pred    = preds["R_val"].squeeze(-1)
    cdir_pred = preds["c_dir_logits"]
    btn_pred  = preds["btn_logits"]

    main_tgt  = torch.stack([targets["main_x"], targets["main_y"]], -1)
    l_tgt     = targets["l_shldr"]
    r_tgt     = targets["r_shldr"]
    cdir_tgt  = targets["c_dir"]
    btn_tgt   = targets.get("btns", targets.get("btns_float")).float()

    loss_main = safe_loss(mse,  main_pred, main_tgt, "main_xy")
    loss_l    = safe_loss(mse,  l_pred,    l_tgt,    "L_val")
    loss_r    = safe_loss(mse,  r_pred,    r_tgt,    "R_val")
    loss_cdir = safe_loss(bce,  cdir_pred, cdir_tgt, "c_dir")
    loss_btn  = safe_loss(bce,  btn_pred,  btn_tgt,  "btns")

    total = loss_main + loss_l + loss_r + loss_cdir + loss_btn
    return total, dict(
        loss_main=loss_main.item(),
        loss_l=loss_l.item(),
        loss_r=loss_r.item(),
        loss_cdir=loss_cdir.item(),
        loss_btn=loss_btn.item(),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset():
    return MeleeFrameDatasetWithDelay(
        parquet_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
    )

def get_dataloader(ds):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

def get_model():
    cfg = ModelConfig(max_seq_len=SEQUENCE_LENGTH)
    model = FramePredictor(cfg).to(DEVICE)
    return model, cfg

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop with detailed logging and checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def train():
    torch.autograd.set_detect_anomaly(True)

    ds = get_dataset()
    dl = get_dataloader(ds)
    model, cfg = get_model()

    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    wandb.init(
        project="FRAME",
        entity="erickfm",
        config=dict(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            num_workers=NUM_WORKERS,
            sequence_length=SEQUENCE_LENGTH,
            reaction_delay=REACTION_DELAY,
            **cfg.__dict__,
        ),
    )

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, batch_ct = 0.0, 0

        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        for i, (state, target) in enumerate(dl, 1):
            for k, v in state.items():
                state[k] = v.to(DEVICE, non_blocking=True)
            for k, v in target.items():
                target[k] = v.to(DEVICE, non_blocking=True)

            preds = model(state)
            loss, metrics = compute_loss(preds, target)

            if i == 1 and epoch == 1:
                grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
                print("=== GRADIENT STATS FOR FIRST BATCH ===")
                for (name, p), g in zip(model.named_parameters(), grads):
                    if g is None:
                        print(f"{name:40s} | no grad")
                    else:
                        n_nan = int(torch.isnan(g).sum().item())
                        n_inf = int(torch.isinf(g).sum().item())
                        norm = float(g.norm().item())
                        print(f"{name:40s} | norm={norm:8.3f}  nan={n_nan:4d}  inf={n_inf:4d}")

            optimiser.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            global_step += 1
            wandb.log(dict(step=global_step, loss=loss.item(), **metrics), step=global_step)

            epoch_loss += loss.item()
            batch_ct += 1

            if i % 25 == 0:
                print(
                    f"[{i:04d}] total={loss.item():.4f} "
                    f"main={metrics['loss_main']:.3f} "
                    f"l={metrics['loss_l']:.3f} "
                    f"r={metrics['loss_r']:.3f} "
                    f"cdir={metrics['loss_cdir']:.3f} "
                    f"btn={metrics['loss_btn']:.3f}"
                )

        avg_loss = epoch_loss / max(batch_ct, 1)
        print(f"Epoch {epoch} done. Avg loss={avg_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/epoch_{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'config': cfg.__dict__,
        }, path)
        print("Saved checkpoint →", path)

        wandb.log(dict(epoch=epoch, avg_loss=avg_loss), step=global_step)

    wandb.finish()

if __name__ == "__main__":
    train()
