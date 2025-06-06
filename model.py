# model.py
# FRAME – Next-frame input prediction model
# -----------------------------------------
# 1. Config
# 2. Causal self-attention & TransformerBlock
# 3. PredictionHeads
# 4. FramePredictor (Encoder ➜ Transformer ➜ Heads)
# 5. Quick sanity check
# -----------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hyper-parameter bundle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # model size
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # sequence
    max_seq_len: int = 64  # longest frame window you’ll pass in (W / T)

    # categorical vocab sizes (match dataset)
    num_stages: int = 32
    num_ports: int = 4
    num_characters: int = 26
    num_actions: int = 88
    num_costumes: int = 6
    num_proj_types: int = 160
    num_proj_subtypes: int = 40


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attention + Transformer block
# ─────────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (no look-ahead)."""
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = x.size(1)
        mask = torch.triu(
            torch.full((W, W), float("-inf"), device=x.device),
            diagonal=1
        )
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class TransformerBlock(nn.Module):
    """(Pre-norm) Causal self-attention → FFN with residuals."""
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
        x = x + self.drop1(self.self_attn(self.norm1(x)))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Output heads
# ─────────────────────────────────────────────────────────────────────────────
class PredictionHeads(nn.Module):
    """
    Outputs:
      - main_xy, c_xy  ∈ [-1,1]
      - L_val, R_val   ∈ [0,1]
      - btn_logits / btn_probs (12-way multi-label)
    """
    def __init__(self, d_model: int, hidden: int = 64, btn_threshold: float = 0.5):
        super().__init__()
        self.btn_threshold = btn_threshold

        def build(out_dim: int, act=None):
            layers = [nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)]
            if act:
                layers.append(act)
            return nn.Sequential(*layers)

        self.main_head = build(2, nn.Tanh())
        self.c_head    = build(2, nn.Tanh())
        self.L_head    = build(1, nn.Sigmoid())
        self.R_head    = build(1, nn.Sigmoid())
        self.btn_head  = build(12)  # raw logits

    def forward(self, h_last: torch.Tensor) -> Dict[str, torch.Tensor]:
        main_xy    = self.main_head(h_last)
        c_xy       = self.c_head(h_last)
        L_val      = self.L_head(h_last)
        R_val      = self.R_head(h_last)
        btn_logits = self.btn_head(h_last)
        btn_probs  = torch.sigmoid(btn_logits)

        return dict(
            main_xy=main_xy,
            c_xy=c_xy,
            L_val=L_val,
            R_val=R_val,
            btn_logits=btn_logits,
            btn_probs=btn_probs,
        )

    def threshold_buttons(self, btn_probs: torch.Tensor) -> torch.Tensor:
        return (btn_probs >= self.btn_threshold).long()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full model: Encoder → Pos-emb → N×Transformer → Heads
# ─────────────────────────────────────────────────────────────────────────────
class FramePredictor(nn.Module):
    """
    Wrapper that:
      1. Encodes raw frame dicts via FrameEncoder → (B,T,d_model)
      2. Adds learned positional embeddings
      3. Runs a stack of causal Transformer blocks
      4. Feeds *last* hidden vector into PredictionHeads
    """
    def __init__(self, cfg: ModelConfig, encoder: Optional[nn.Module] = None):
        super().__init__()
        # Local import avoids circular dependency if FrameEncoder imports model.py
        from frame_encoder import FrameEncoder

        # NOTE: no `dropout` or `d_model` kwargs – matches your encoder’s signature
        self.encoder = encoder or FrameEncoder(
            num_stages=cfg.num_stages,
            num_ports=cfg.num_ports,
            num_characters=cfg.num_characters,
            num_actions=cfg.num_actions,
            num_costumes=cfg.num_costumes,
            num_proj_types=cfg.num_proj_types,
            num_proj_subtypes=cfg.num_proj_subtypes,
        )

        self.pos_emb = nn.Parameter(
            torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.heads = PredictionHeads(cfg.d_model)

    def forward(self, frames: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.encoder(frames)                # (B,T,d_model)
        W = x.size(1)
        x = x + self.pos_emb[:, :W]

        for blk in self.blocks:
            x = blk(x)

        h_last = x[:, -1]                       # (B,d_model)
        return self.heads(h_last)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = ModelConfig()
    model = FramePredictor(cfg)

    B, T = 2, 32
    rand_int   = lambda high: torch.randint(0, high, (B, T))
    rand_float = lambda d: torch.randn(B, T, d)

    dummy_frames = {
        # categorical
        "stage":              rand_int(cfg.num_stages),
        "self_port":          rand_int(cfg.num_ports),
        "opp_port":           rand_int(cfg.num_ports),
        "self_character":     rand_int(cfg.num_characters),
        "opp_character":      rand_int(cfg.num_characters),
        "self_action":        rand_int(cfg.num_actions),
        "opp_action":         rand_int(cfg.num_actions),
        "self_costume":       rand_int(cfg.num_costumes),
        "opp_costume":        rand_int(cfg.num_costumes),
        "self_nana_character":rand_int(cfg.num_characters),
        "opp_nana_character": rand_int(cfg.num_characters),
        "self_nana_action":   rand_int(cfg.num_actions),
        "opp_nana_action":    rand_int(cfg.num_actions),
    }

    # projectile categorical
    for j in range(8):
        dummy_frames[f"proj{j}_owner"]   = rand_int(cfg.num_ports)
        dummy_frames[f"proj{j}_type"]    = rand_int(cfg.num_proj_types)
        dummy_frames[f"proj{j}_subtype"] = rand_int(cfg.num_proj_subtypes)

    # numeric / boolean
    dummy_frames.update({
        "numeric":             rand_float(20),
        "self_numeric":        rand_float(22),
        "opp_numeric":         rand_float(22),
        "self_nana_numeric":   rand_float(27),
        "opp_nana_numeric":    rand_float(27),

        "self_analog":         rand_float(6),
        "opp_analog":          rand_float(6),
        "self_nana_analog":    rand_float(6),
        "opp_nana_analog":     rand_float(6),

        # per-slot numeric (B,T,5)  ✅
        **{f"{k}_numeric": rand_float(5) for k in map(str, range(8))},

        # buttons / flags
        "self_buttons":        torch.randint(0, 2, (B, T, 12)).bool(),
        "opp_buttons":         torch.randint(0, 2, (B, T, 12)).bool(),
        "self_flags":          torch.randint(0, 2, (B, T, 5)).bool(),
        "opp_flags":           torch.randint(0, 2, (B, T, 5)).bool(),
        "self_nana_buttons":   torch.randint(0, 2, (B, T, 12)).bool(),
        "opp_nana_buttons":    torch.randint(0, 2, (B, T, 12)).bool(),
        "self_nana_flags":     torch.randint(0, 2, (B, T, 6)).bool(),
        "opp_nana_flags":      torch.randint(0, 2, (B, T, 6)).bool(),
    })

    with torch.no_grad():
        out = model(dummy_frames)

    for k, v in out.items():
        print(f"{k:10s} {tuple(v.shape)}")
