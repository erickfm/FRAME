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
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Training hyperparameters
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

    # stack input frames
    for k in batch[0][0].keys():
        batch_state[k] = torch.stack([item[0][k] for item in batch], dim=0)
    # stack target frames
    for k in batch[0][1].keys():
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

    main_target = torch.stack([targets["main_x"], targets["main_y"]], dim=-1)
    c_target    = torch.stack([targets["c_x"], targets["c_y"]], dim=-1)
    l_target    = targets["l_shldr"]
    r_target    = targets["r_shldr"]
    btn_target  = targets.get("btns", targets.get("btns_float")).float()

    loss_main = mse(main_pred, main_target)
    loss_c    = mse(c_pred, c_target)
    loss_l    = mse(l_pred, l_target)
    loss_r    = mse(r_pred, r_target)
    loss_btn  = bce(btn_pred, btn_target)

    total = (10 * loss_main) + (10 * loss_c) + loss_l + loss_r + loss_btn
    return total, {
        "loss_main": loss_main.item(),
        "loss_c":    loss_c.item(),
        "loss_l":    loss_l.item(),
        "loss_r":    loss_r.item(),
        "loss_btn":  loss_btn.item(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. Data and model setup
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset():
    return MeleeFrameDatasetWithDelay(
        parquet_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        reaction_delay=REACTION_DELAY,
    )


def get_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )


def get_model():
    # use canonical vocab sizes from cat_maps
    cfg = ModelConfig(max_seq_len=SEQUENCE_LENGTH)
    model = FramePredictor(cfg).to(DEVICE)
    return model, cfg

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    dataset   = get_dataset()
    dataloader= get_dataloader(dataset)
    model, cfg= get_model()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # persist config in checkpoint and log to wandb
    wandb.init(
        project="FRAME",
        entity="erickfm",
        config={
            # training
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "num_workers": NUM_WORKERS,
            "sequence_length": SEQUENCE_LENGTH,
            "reaction_delay": REACTION_DELAY,
            "rollout_steps": ROLL_OUT_STEPS,
            # model config
            **cfg.__dict__,
        }
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batch_count= 0

        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===\n")
        for i, (batch_state, batch_target) in enumerate(dataloader, start=1):
            # move to device
            for k,v in batch_state.items(): batch_state[k]= v.to(DEVICE)
            for k,v in batch_target.items(): batch_target[k]= v.to(DEVICE)

            preds, loss_dict = model(batch_state), None
            loss, loss_metrics = compute_loss(preds, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if i % 25 == 0:
                print(
                    f"[Batch {i}] Loss={loss.item():.4f} "
                    f"(main={loss_metrics['loss_main']:.3f}, c={loss_metrics['loss_c']:.3f}, "
                    f"l={loss_metrics['loss_l']:.3f}, r={loss_metrics['loss_r']:.3f}, "
                    f"btn={loss_metrics['loss_btn']:.3f})"
                )

        avg_loss = epoch_loss / batch_count
        print(f"\nEpoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint (with config)
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
        }
        ckpt_path = f"checkpoints/epoch_{epoch:02d}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            **loss_metrics,
            "checkpoint": ckpt_path,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
