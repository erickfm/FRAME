# sanity_check.py
from pathlib import Path
import torch

from dataset import MeleeFrameDatasetWithDelay
from model import FramePredictor, derive_config_from_dataset

# 1) Build the dataset (point to a few small test parquets to start)
data_dir = Path("./data")          # adjust as needed
ds = MeleeFrameDatasetWithDelay(
    parquet_dir=data_dir,
    sequence_length=30,
    reaction_delay=1,
)

# 2) Derive a ModelConfig that matches ds’s categorical vocab sizes
cfg = derive_config_from_dataset(ds)

# 3) Instantiate the predictor
model = FramePredictor(cfg)
model.eval()                       # disable dropout for the check

# 4) Grab a mini-batch from the dataset
batch_size = 2
idxs = torch.randint(0, len(ds), (batch_size,))
frames, targets = zip(*(ds[i] for i in idxs))

# Stack dicts → dict of tensors (B,T,…)
def stack_dicts(samples):
    out = {}
    for k in samples[0]:
        out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out

frames = stack_dicts(frames)

# 5) Forward pass
with torch.no_grad():
    preds = model(frames)

# 6) Print shapes
print("=== Sanity check shapes ===")
for name, tensor in preds.items():
    print(f"{name:10s} {tuple(tensor.shape)}")

# 7) Optional quick loss check (MSE on analog sticks)
mse = torch.nn.functional.mse_loss(
    preds["main_xy"],
    torch.stack([t["main_x"].view(-1, 1).repeat(1, 2)  # crude reshape just for demo
                 for t in targets], dim=0)
)
print("Dummy MSE (main_xy vs. target main_x):", mse.item())
