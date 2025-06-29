# frame_encoder.py – intra‑frame cross‑attention version
# -----------------------------------------------------------------------------
# Converts structured Melee frame dictionaries -> (B, T, d_model) tensor.
# Compared to the old "flat‑concat" encoder, this version:
#   1. Builds **one token per feature‑group** (stage, sticks, buttons …)
#   2. Runs 1–2 layers of **self‑attention across those tokens** *inside the frame*
#   3. Pools via a learned `[CLS]` token → single 256‑d summary vector
#   4. Projects up to `d_model` (1024) so it plugs into the temporal Transformer
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Hyper‑params / helpers
# -----------------------------------------------------------------------------
DROPOUT_P = 0.10          # global dropout applied everywhere
D_INTRA   = 256           # width for intra‑frame tokens


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """LayerNorm → Linear → GELU → Dropout."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Dropout(DROPOUT_P),
    )


# -----------------------------------------------------------------------------
# Intra‑frame pooling via a mini Transformer (a.k.a Set Attention / PMA)
# -----------------------------------------------------------------------------
class _GroupAttention(nn.Module):
    """Self‑attention across the G feature‑group tokens inside **one** frame."""

    def __init__(self, d_intra: int = D_INTRA, nhead: int = 4, nlayers: int = 2, k_query: int = 1):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_intra,
            nhead=nhead,
            dim_feedforward=4 * d_intra,
            dropout=DROPOUT_P,
            batch_first=True,  # (B, N, C)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)

        # k learnable query tokens (k=1 → classic [CLS])
        self.k_query = k_query
        self.queries = nn.Parameter(torch.randn(k_query, d_intra) * 0.02)

    def forward(self, group_tokens: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        group_tokens : Tensor
            (B*T, G, d_intra) stacked tokens for one frame per batch‑time example.

        Returns
        -------
        Tensor
            (B*T, d_intra) pooled representation.
        """
        BxT = group_tokens.size(0)
        k = self.k_query
        # Insert learned queries at the front: shape (B*T, k+G, d_intra)
        q = self.queries.unsqueeze(0).expand(BxT, -1, -1)
        x = torch.cat([q, group_tokens], dim=1)
        h = self.encoder(x)        # full self‑attention, no mask

        if k == 1:
            pooled = h[:, 0]       # (B*T, d_intra)
        else:
            # mean the k query outputs (≈ PMA pooling)
            pooled = h[:, :k].mean(dim=1)
        return pooled


# -----------------------------------------------------------------------------
# FrameEncoder – cross‑attention version
# -----------------------------------------------------------------------------
class FrameEncoder(nn.Module):
    """Encode per‑frame structured dict → (B, T, d_model) using intra‑frame attention."""

    # ------- Constants describing raw numeric dims from the dataset ---------
    GLOBAL_NUM   = 20
    PLAYER_NUM   = 22
    NANA_NUM     = 27
    ANALOG_DIM   = 4      # main_x, main_y, l, r
    BTN_DIM      = 12
    FLAGS_DIM    = 5
    NANA_FLAGS   = 6
    PROJ_NUM_PER = 5
    PROJ_SLOTS   = 8

    def __init__(
        self,
        *,
        num_stages: int,
        num_ports: int,
        num_characters: int,
        num_actions: int,
        num_costumes: int,
        num_proj_types: int,
        num_proj_subtypes: int,
        d_model: int = 1024,
        num_c_dirs: int = 5,
    ) -> None:
        super().__init__()

        # -------------- Embeddings / per‑group encoders ----------------------
        # Categorical → Emb + Linear → d_intra
        def cat_block(n_vocab: int) -> nn.Module:
            return nn.Sequential(
                nn.Embedding(n_vocab, D_INTRA),
                nn.Dropout(DROPOUT_P),
            )

        self.stage_emb      = cat_block(num_stages)
        self.port_emb       = cat_block(num_ports)
        self.char_emb       = cat_block(num_characters)
        self.act_emb        = cat_block(num_actions)
        self.cost_emb       = cat_block(num_costumes)
        self.cdir_emb       = cat_block(num_c_dirs)
        self.ptype_emb      = cat_block(num_proj_types)
        self.psub_emb       = cat_block(num_proj_subtypes)

        # Numeric / boolean groups → small MLP → d_intra
        self.glob_enc       = _mlp(self.GLOBAL_NUM, D_INTRA)
        self.player_enc     = _mlp(self.PLAYER_NUM + 1, D_INTRA)       # + elapsed
        self.nana_enc       = _mlp(self.NANA_NUM + 1, D_INTRA)
        self.analog_enc     = _mlp(self.ANALOG_DIM, D_INTRA)
        self.proj_num_enc   = _mlp(self.PROJ_NUM_PER * self.PROJ_SLOTS, D_INTRA)
        self.btn_enc        = _mlp(self.BTN_DIM * 2, D_INTRA)
        self.flag_enc       = _mlp(self.FLAGS_DIM * 2, D_INTRA)
        self.nana_btn_enc   = _mlp(self.BTN_DIM * 2, D_INTRA)
        self.nana_flag_enc  = _mlp(self.NANA_FLAGS * 2, D_INTRA)

        # -------------- Intra‑frame attention block -------------------------
        self.set_attn = _GroupAttention(d_intra=D_INTRA, nhead=4, nlayers=2, k_query=1)

        # -------------- Final projection to d_model -------------------------
        self.to_model = nn.Sequential(
            nn.LayerNorm(D_INTRA),
            nn.Linear(D_INTRA, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
        )

    # --------------------------- helper fns --------------------------------
    @staticmethod
    def _collapse(x: torch.Tensor) -> torch.Tensor:  # (B,T,*) -> (B*T, *)
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:])

    # --------------------------- forward -----------------------------------
    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a window of frames.

        Parameters
        ----------
        seq : Mapping[str, Tensor]
            All tensors already shaped (B, T, …).

        Returns
        -------
        Tensor
            (B, T, d_model) ready for the temporal Transformer.
        """
        B, T = seq["stage"].shape  # batch, time

        # ---------------- Per‑group tokens (B,T,d_intra) -------------------
        tok: List[torch.Tensor] = []
        def add(x: torch.Tensor):
            tok.append(x)

        # — categorical groups —
        add(self.stage_emb(seq["stage"]))
        add(self.port_emb(seq["self_port"]))
        add(self.port_emb(seq["opp_port"]))
        add(self.char_emb(seq["self_character"]))
        add(self.char_emb(seq["opp_character"]))
        add(self.act_emb(seq["self_action"]))
        add(self.act_emb(seq["opp_action"]))
        add(self.cost_emb(seq["self_costume"]))
        add(self.cost_emb(seq["opp_costume"]))
        add(self.cdir_emb(seq["self_c_dir"]))
        add(self.cdir_emb(seq["opp_c_dir"]))
        # — nana cats —
        add(self.char_emb(seq["self_nana_character"]))
        add(self.char_emb(seq["opp_nana_character"]))
        add(self.act_emb(seq["self_nana_action"]))
        add(self.act_emb(seq["opp_nana_action"]))
        add(self.cdir_emb(seq["self_nana_c_dir"]))
        add(self.cdir_emb(seq["opp_nana_c_dir"]))
        # — projectile cats —
        for j in range(self.PROJ_SLOTS):
            add(self.port_emb(seq[f"proj{j}_owner"]))
            add(self.ptype_emb(seq[f"proj{j}_type"]))
            add(self.psub_emb(seq[f"proj{j}_subtype"]))

        # — numeric / boolean groups —
        add(self.glob_enc(seq["numeric"]))
        add(self.player_enc(torch.cat([seq["self_numeric"], seq["self_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        add(self.player_enc(torch.cat([seq["opp_numeric"], seq["opp_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        add(self.nana_enc(torch.cat([seq["self_nana_numeric"], seq["self_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        add(self.nana_enc(torch.cat([seq["opp_nana_numeric"], seq["opp_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1)))
        add(self.analog_enc(seq["self_analog"]))
        add(self.analog_enc(seq["opp_analog"]))
        add(self.analog_enc(seq["self_nana_analog"]))
        add(self.analog_enc(seq["opp_nana_analog"]))
        add(self.proj_num_enc(torch.cat([seq[f"{k}_numeric"] for k in map(str, range(self.PROJ_SLOTS))], dim=-1)))
        add(self.btn_enc(torch.cat([seq["self_buttons"].float(), seq["opp_buttons"].float()], dim=-1)))
        add(self.flag_enc(torch.cat([seq["self_flags"].float(), seq["opp_flags"].float()], dim=-1)))
        add(self.nana_btn_enc(torch.cat([seq["self_nana_buttons"].float(), seq["opp_nana_buttons"].float()], dim=-1)))
        add(self.nana_flag_enc(torch.cat([seq["self_nana_flags"].float(), seq["opp_nana_flags"].float()], dim=-1)))

        # Stack along new token dim → (B,T,G,d_intra)
        group_stack = torch.stack(tok, dim=2)      # G = len(tok)

        # Collapse (B,T) → (B*T) for per‑frame attention
        group_flat = self._collapse(group_stack)   # (BxT, G, d_intra)

        # Intra‑frame attention + pooling
        pooled_flat = self.set_attn(group_flat)    # (BxT, d_intra)

        # Restore (B,T, d_intra)
        pooled = pooled_flat.view(B, T, D_INTRA)

        # Project to model width
        return self.to_model(pooled)               # (B, T, d_model)
