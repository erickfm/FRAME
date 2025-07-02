#!/usr/bin/env python3
# train.py  —  FRAME training with strict finiteness checks
#             + AdamW, GradNorm, and fp16 AMP (PyTorch ≥ 2.3)

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader

# —— AMP (new namespace in ≥2.3) ————————————————————————————————
from torch.amp import autocast, GradScaler

import wandb

from dataset import MeleeFrameDataset
from model   import FramePredictor, ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE      = 256
NUM_EPOCHS      = 200
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-2            # AdamW (weights only)
NUM_WORKERS     = 16
SEQUENCE_LENGTH = 60
REACTION_DELAY  = 1
DATA_DIR        = "./data"

GRAD_CLIP_NORM  = 1.0
GRADNORM_ALPHA  = 1.0
TASK_NAMES      = ["main", "l", "r", "cdir", "btn"]

USE_AMP         = torch.cuda.is_available()  # fp16 only if CUDA

# ─────────────────────────────────────────────────────────────────────────────
# 2. Collate
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    batch_state, batch_target = {}, {}
    for k in batch[0][0]:
        batch_state[k]  = torch.stack([item[0][k] for item in batch], 0)
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
    # —— loss fns ————————————————————————————————————————————————
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss(reduction='mean')
    bce = nn.BCEWithLogitsLoss()

    # —— predictions (cast to fp32 so dtypes match targets) ————————
    main_pred = preds["main_xy"].float()
    l_pred    = preds["L_val"].squeeze(-1).float()
    r_pred    = preds["R_val"].squeeze(-1).float()
    c_logits  = preds["c_dir_logits"].float()
    btn_pred  = preds["btn_logits"].float()

    # —— targets ————————————————————————————————————————————————
    main_tgt = torch.stack([targets["main_x"], targets["main_y"]], dim=-1)
    l_tgt    = targets["l_shldr"]
    r_tgt    = targets["r_shldr"]
    cdir_tgt = targets["c_dir"].long()
    btn_tgt  = targets.get("btns", targets.get("btns_float")).float()

    # —— per-head losses ————————————————————————————————————————
    loss_main = safe_loss(mse, main_pred, main_tgt, "main_xy")
    loss_l    = safe_loss(mse, l_pred,    l_tgt,    "L_val")
    loss_r    = safe_loss(mse, r_pred,    r_tgt,    "R_val")
    loss_cdir = safe_loss(ce,  c_logits,  cdir_tgt.argmax(dim=-1), "c_dir")
    loss_btn  = safe_loss(bce, btn_pred,  btn_tgt, "btns")

    metrics = {
        "loss_main": loss_main.item(),
        "loss_l":    loss_l.item(),
        "loss_r":    loss_r.item(),
        "loss_cdir": loss_cdir.item(),
        "loss_btn":  loss_btn.item(),
    }
    task_losses = (loss_main, loss_l, loss_r, loss_cdir, loss_btn)
    return metrics, task_losses

# ─────────────────────────────────────────────────────────────────────────────
# 4. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset():
    return MeleeFrameDataset(
        parquet_dir=DATA_DIR,
        seq_len=SEQUENCE_LENGTH,
        delay=REACTION_DELAY,
    )

def get_dataloader(ds):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

def get_model():
    cfg   = ModelConfig(max_seq_len=SEQUENCE_LENGTH)
    model = FramePredictor(cfg).to(DEVICE)
    return model, cfg

# ─────────────────────────────────────────────────────────────────────────────
# CLI / logging / resume setup
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Realtime FRAME bot w/ debug")
parser.add_argument("--debug",  action="store_true", help="Verbose sanity checks")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint (.pt) to resume from")
parser.add_argument("--no-gradnorm", action="store_true",
                    help="Disable GradNorm loss balancing")
args  = parser.parse_args()
DEBUG = args.debug or bool(os.getenv("DEBUG", ""))
USE_GRADNORM = not args.no_gradnorm

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop with sanity checks, GradNorm, AMP, checkpointing
# ─────────────────────────────────────────────────────────────────────────────
# --- constants for GradNorm clamping -----------------------------------------
CDIR_IDX   = TASK_NAMES.index("cdir")  # make sure this matches your TASK_NAMES
MIN_W_CDIR = 0.7

def train():
    torch.autograd.set_detect_anomaly(True)

    ds, dl       = get_dataset(), get_dataloader(get_dataset())
    model, cfg   = get_model()

    # —— GradNorm learnable weights ——————————————————————————————
    loss_weights = torch.nn.Parameter(torch.ones(len(TASK_NAMES), device=DEVICE))
    if not USE_GRADNORM:
        loss_weights.requires_grad_(False)

    # —— AdamW param-groups (bias/Norm excluded from decay) ————————
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith("bias") or "norm" in n.lower()
                  else decay).append(p)

    optim_groups = [
        {"params": decay,    "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if USE_GRADNORM:
        optim_groups.append({"params": [loss_weights], "weight_decay": 0.0})
    optimiser = optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.999))

    scaler = GradScaler(enabled=USE_AMP)

    # —— optional resume ————————————————————————————————————————
    start_epoch    = 1
    init_task_loss = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimiser.load_state_dict(ckpt['optimizer_state_dict'])
        if USE_GRADNORM and 'loss_weights' in ckpt:
            loss_weights.data.copy_(ckpt['loss_weights'])
        init_task_loss = ckpt.get("init_task_loss")
        if USE_GRADNORM and init_task_loss is not None:
            init_task_loss = init_task_loss.to(DEVICE)
        if USE_AMP and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt['epoch'] + 1

    wandb.init(
        project="FRAME", entity="erickfm",
        config=dict(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                    epochs=NUM_EPOCHS, num_workers=NUM_WORKERS,
                    sequence_length=SEQUENCE_LENGTH, reaction_delay=REACTION_DELAY,
                    amp=USE_AMP, **cfg.__dict__),
    )

    global_step = 0
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, batch_ct = 0.0, 0
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

        for i, (state, target) in enumerate(dl, 1):
            # move to device & sanity-check
            for k, v in {**state, **target}.items():
                v = v.to(DEVICE, non_blocking=True)
                if DEBUG and not torch.isfinite(v).all():
                    raise RuntimeError(f"Non-finite tensor in {k}")
                if k in state: state[k] = v
                else:         target[k] = v

            # forward
            with autocast("cuda", enabled=USE_AMP):
                preds = model(state)

            # compute losses + GradNorm
            metrics, task_losses = compute_loss(preds, target)
            loss_vec = torch.stack(task_losses)

            if USE_GRADNORM:
                if init_task_loss is None:
                    init_task_loss = loss_vec.detach()

                weighted = loss_weights * loss_vec
                task_loss = weighted.sum()

                avg_loss = loss_vec.mean().detach()
                inv_rate = (loss_vec / init_task_loss).detach()
                target_g = avg_loss * inv_rate.pow(GRADNORM_ALPHA)
                gradnorm = nn.functional.l1_loss(loss_vec, target_g)

                total_loss = task_loss + gradnorm
            else:
                task_loss = loss_vec.sum()
                gradnorm = torch.tensor(0.0, device=DEVICE)
                total_loss = task_loss

            # optional debug grads on first batch
            if i == 1 and epoch == start_epoch:
                grads = torch.autograd.grad(total_loss, model.parameters(), retain_graph=True)
                print("=== FIRST-BATCH GRADS ===")
                for (n, p), g in zip(model.named_parameters(), grads):
                    if g is None:
                        print(f"{n:40s} | no grad")
                    else:
                        print(f"{n:40s} | norm={g.norm():.3f}")

            # backward + step
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimiser)
            nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimiser)
            scaler.update()

            # —— clamp cdir weight floor ——————————————————————
            if USE_GRADNORM:
                with torch.no_grad():
                    # keep all weights ≥ 0
                    loss_weights.data.clamp_(min=0.0)
                    # but force cdir ≥ MIN_W_CDIR
                    loss_weights.data[CDIR_IDX].clamp_(min=MIN_W_CDIR)

            # logging
            global_step += 1
            log_data = {
                "step": global_step,
                "total": total_loss.item(),
                "gradnorm": gradnorm.item(),
                **metrics,
            }
            if USE_GRADNORM:
                log_data.update({f"w_{n}": loss_weights[j].item() for j, n in enumerate(TASK_NAMES)})
            wandb.log(log_data, step=global_step)

            epoch_loss += total_loss.item()
            batch_ct   += 1
            if i % 25 == 0:
                print(
                    f"[{i:04d}] total={total_loss:.4f} "
                    f"main={metrics['loss_main']:.3f} "
                    f"cdir={metrics['loss_cdir']:.3f} gn={gradnorm:.3f}"
                )

        # end epoch checkpoint
        avg = epoch_loss / max(batch_ct, 1)
        print(f"Epoch {epoch} done. Avg loss={avg:.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "loss_weights":         loss_weights.data.cpu(),
            "init_task_loss":       init_task_loss.cpu() if init_task_loss is not None else None,
            "scaler_state_dict":    scaler.state_dict(),
            "config":               cfg.__dict__,
        }, ckpt_path)
        print("Saved checkpoint →", ckpt_path)
        wandb.log({"epoch": epoch, "avg_loss": avg}, step=global_step)

    wandb.finish()


if __name__ == "__main__":
    train()
