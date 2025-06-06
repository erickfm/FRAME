import torch
from dataset import MeleeFrameDatasetWithDelay
from model import FramePredictor, ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Dataset config (same as in train.py)
dataset = MeleeFrameDatasetWithDelay(
    parquet_dir="./data",
    sequence_length=30,
    reaction_delay=6,
)
print(f"Loaded {len(dataset)} frame sequences.")

# Get a single sample
frame_dict, target = dataset[0]
for k in frame_dict:
    print(f"{k:20s}: {frame_dict[k].shape}  dtype={frame_dict[k].dtype}")
for k in target:
    print(f"{k:20s}: {target[k].shape}  dtype={target[k].dtype}")

# Batchify (1 item → 1 batch)
for k in frame_dict:
    frame_dict[k] = frame_dict[k].unsqueeze(0)  # [1, W] or [1, W, D]
for k in target:
    target[k] = target[k].unsqueeze(0)          # [1, D]

# Load model
cfg = ModelConfig(
    max_seq_len=30,
    num_stages=32,
    num_ports=4,
    num_characters=26,
    num_actions=88,
    num_costumes=6,
    num_proj_types=160,
    num_proj_subtypes=40,
)
model = FramePredictor(cfg).to(DEVICE)
model.eval()

# Move to device
frame_dict = {k: v.to(DEVICE) for k, v in frame_dict.items()}

# Run forward pass
with torch.no_grad():
    out = model(frame_dict)

print("\nPredicted output:")
for k, v in out.items():
    print(f"{k:12s}: {tuple(v.shape)}")
