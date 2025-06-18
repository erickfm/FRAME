# FRAME: Fixed-step Realtime Action Matrix Estimator
FRAME is a decoder-only transformer model for predicting next-frame inputs in Super Smash Bros. Melee. Built around fixed-step causal attention, it estimates future controller states from sequences of gameplay data.


## Features
- Decoder-only architecture (GPT-like)

- Supports real-time inference on live frame sequences

- Trained on Slippi-derived datasets

- Predicts analog stick, C-stick, trigger, and button inputs


## Default configuration
| setting                | value                                       |
|------------------------|---------------------------------------------|
| Window length (W)      | 30 frames                                   |
| Roll-out steps (R)     | 8 frames                                    |
| Hidden size            | 256                                         |
| Transformer layers     | 2                                           |
| Attention heads        | 8                                           |
| Embedding dims         | stage 32, char 32, action 32, projectile 16 |


## Project Structure
```
.
├── train.py          # Training loop & checkpointing
├── eval.py           # Validation / quick metrics
├── inference.py      # Real-time single-window inference
├── model.py          # FrameEncoder + Transformer heads
├── dataset.py        # Parquet → tensor windows
├── utils.py          # Normalization, button mapping, etc.
├── checkpoints/      # Saved *.pt files
└── data/             # Place Slippi parquet files here
```

## Setup

Install dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```
