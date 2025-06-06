import os
import time
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MeleeFrameDatasetWithDelay
from model import FramePredictor, ModelConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE      = 128
NUM_EPOCHS      = 10
LEARNING_RATE   = 3e-4
NUM_WORKERS     = 4
SEQUENCE_LENGTH = 30
REACTION_DELAY  = 1
ROLL_OUT_STEPS  = 1
DATA_DIR        = "./data"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Collate function
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    batch_state = {}
    batch_target = {}

    state_keys = batch[0][0].keys()
    for k in state_keys:
        batch_state[k] = torch.stack([item[0][k] for item in batch], dim=0)

    target_keys = batch[0][1].keys()
    for k in target_keys:
        batch_target[k] = torch.stack([item[1][k] for item in batch], dim=0)

    return batch_state, batch_target

# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss function
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(preds, targets):
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    main_pred = preds["main_xy"]
    c_pred    = preds["c_xy"]
    l_pred    = preds["L_val"].squeeze(-1)
    r_pred    = preds["R_val"].squeeze(-1)
    btn_pred  = preds["btn_logits"]

    main_target = torch.stack([targets["main_x"], targets["main_y"]], dim=1)
    c_target    = torch.stack([targets["c_x"], targets["c_y"]], dim=1)
    l_target    = targets["l_shldr"]
    r_target    = targets["r_shldr"]
    btn_target  = targets["btns"].float()

    loss_main = mse(main_pred, main_target)
    loss_c    = mse(c_pred, c_target)
    loss_l    = mse(l_pred, l_target)
    loss_r    = mse(r_pred, r_target)
    loss_btn  = bce(btn_pred, btn_target)

    total = loss_main + loss_c + loss_l + loss_r + loss_btn
    return total, {
        "loss_main": loss_main.item(),
        "loss_c":    loss_c.item(),
        "loss_l":    loss_l.item(),
        "loss_r":    loss_r.item(),
        "loss_btn":  loss_btn.item(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. Setup
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloader():
    dataset = MeleeFrameDatasetWithDelay(
        parquet_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )

def get_model():
    cfg = ModelConfig(
        max_seq_len=SEQUENCE_LENGTH,
        num_stages=32,
        num_ports=4,
        num_characters=26,
        num_actions=88,
        num_costumes=6,
        num_proj_types=160,
        num_proj_subtypes=40,
    )
    return FramePredictor(cfg).to(DEVICE), cfg

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    dataloader = get_dataloader()
    model, cfg = get_model()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    wandb.init(project="FRAME", entity="erickfm", config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "sequence_length": SEQUENCE_LENGTH,
        "reaction_delay": REACTION_DELAY,
        "rollout_steps": ROLL_OUT_STEPS,
        "model_dim": cfg.d_model,
    })

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===\n")
        for i, (batch_state, batch_target) in enumerate(dataloader, 1):
            for k in batch_state:
                batch_state[k] = batch_state[k].to(DEVICE)
            for k in batch_target:
                batch_target[k] = batch_target[k].to(DEVICE)

            preds = model(batch_state)
            loss, loss_dict = compute_loss(preds, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if i % 25 == 0:
                print(f"[Batch {i}] Loss = {loss.item():.4f} "
                      f"(main={loss_dict['loss_main']:.3f}, "
                      f"c={loss_dict['loss_c']:.3f}, "
                      f"l={loss_dict['loss_l']:.3f}, "
                      f"r={loss_dict['loss_r']:.3f}, "
                      f"btn={loss_dict['loss_btn']:.3f})")

        avg = epoch_loss / batch_count
        print(f"\nEpoch {epoch} complete. Avg Loss: {avg:.4f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = f"checkpoints/epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt)
        print(f"Saved checkpoint to {ckpt}")

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg,
            **loss_dict,
            "checkpoint": ckpt,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
