#!/usr/bin/env python3
# model.py ─ FRAME next-frame predictor
# ------------------------------------
# 1. Hyper-parameter bundle
# 2. Causal self-attention & TransformerBlock
# 3. PredictionHeads
# 4. FramePredictor (Encoder ➜ Transformer ➜ Heads)
# ------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

# Source-of-truth enum maps
from cat_maps import (
    STAGE_MAP,
    CHARACTER_MAP,
    ACTION_MAP,
    PROJECTILE_TYPE_MAP,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Hyper-parameter bundle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # model size (tiny “overfit” test)
    d_model: int         = 64      # was 512
    nhead: int           = 4       # d_model must be divisible by nhead
    num_layers: int      = 1       # was 4
    dim_feedforward: int = 128     # was 1024
    dropout: float       = 0.0     # turn off dropout to help overfit

    # sequence length
    max_seq_len: int     = 60      # was 120

    # fixed categorical vocab sizes (keep your real maps)
    num_stages: int       = len(STAGE_MAP)
    num_ports: int        = 4
    num_characters: int   = len(CHARACTER_MAP)
    num_actions: int      = len(ACTION_MAP)
    num_costumes: int     = 6
    num_c_dirs: int       = 5
    num_proj_types: int   = len(PROJECTILE_TYPE_MAP)
    num_proj_subtypes: int= 40

# ─────────────────────────────────────────────────────────────────────────────
# 2. Attention + Transformer block
# ─────────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an upper-triangular (causal) mask."""
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = x.shape[1]
        mask = torch.triu(
            torch.full((W, W), float("-inf"), device=x.device),
            diagonal=1,
        )
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with causal self-attention."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(cfg.d_model, cfg.nhead, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.drop1 = nn.Dropout(cfg.dropout)

        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.self_attn(self.norm1(x)))  # attention
        x = x + self.drop2(self.ff(self.norm2(x)))         # feed-forward
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Output heads
# ─────────────────────────────────────────────────────────────────────────────
class PredictionHeads(nn.Module):
    """
    Outputs per next-frame target:
      - main_xy          ∈ [-1, 1]              (regression, MSE/L1)
      - L_val, R_val     ∈ [0, 1]               (regression, MSE)
      - c_dir_logits  5-way logits              (classification, BCE w/ one-hot)
      - btn_logits       12-way logits          (multi-label BCE)
    """
    def __init__(self, d_model: int, hidden: int = 64, btn_threshold: float = 0.5):
        super().__init__()
        self.btn_threshold = btn_threshold

        def build(out_dim: int, activation: Optional[nn.Module] = None):
            layers = [nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)]
            if activation:
                layers.append(activation)
            return nn.Sequential(*layers)

        self.main_head  = build(2, nn.Tanh())
        self.L_head     = build(1, nn.Sigmoid())
        self.R_head     = build(1, nn.Sigmoid())
        self.cdir_head  = build(5)                # raw logits
        self.btn_head   = build(12)               # raw logits

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        main_xy      = self.main_head(h)
        L_val        = self.L_head(h)
        R_val        = self.R_head(h)
        c_dir_logits = self.cdir_head(h)
        btn_logits   = self.btn_head(h)

        return dict(
            main_xy=main_xy,
            L_val=L_val,
            R_val=R_val,
            c_dir_logits=c_dir_logits,
            c_dir_probs=torch.sigmoid(c_dir_logits),
            btn_logits=btn_logits,
            btn_probs=torch.sigmoid(btn_logits),
        )

    def threshold_buttons(self, btn_probs: torch.Tensor) -> torch.Tensor:
        return btn_probs >= self.btn_threshold


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full model: Encoder → Pos-emb → N×Transformer → Heads
# ─────────────────────────────────────────────────────────────────────────────
class FramePredictor(nn.Module):
    """
    1. Encodes structured frames via FrameEncoder → (B, T, d_model)
    2. Adds learned positional embeddings
    3. Runs a stack of causal Transformer blocks
    4. Feeds the final hidden vector into PredictionHeads
    """
    def __init__(self, cfg: ModelConfig, encoder: Optional[nn.Module] = None):
        super().__init__()
        # avoid circular import
        from frame_encoder import FrameEncoder

        self.cfg = cfg
        self.encoder = encoder or FrameEncoder(
            num_stages=cfg.num_stages,
            num_ports=cfg.num_ports,
            num_characters=cfg.num_characters,
            num_actions=cfg.num_actions,
            num_costumes=cfg.num_costumes,
            num_proj_types=cfg.num_proj_types,
            num_proj_subtypes=cfg.num_proj_subtypes,
            d_model=cfg.d_model,
            num_c_dirs=cfg.num_c_dirs,
        )

        # learned positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        # transformer stack
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])

        # final normalization for stability
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # prediction heads
        self.heads = PredictionHeads(cfg.d_model)

    def forward(self, frames: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.encoder(frames)                         # (B, T, d_model)
        T = x.size(1)
        x = x + self.pos_emb[:, :T]                      # add positional info
        for blk in self.blocks:
            x = blk(x)
        h_last = x[:, -1]                                # final step
        h_last = self.final_norm(h_last)                # normalize for stability
        return self.heads(h_last)
