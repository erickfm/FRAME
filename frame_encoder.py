# frame_encoder.py – intra‑frame cross‑attention (Composite‑Token v2, verbose comments)
# -----------------------------------------------------------------------------
# Converts one structured Melee frame (already batched as tensors) → (B, T, d_model)
# tensor that a *temporal* Transformer can ingest.  Design philosophy mirrors the
# original “v1” version you liked, but we adopt the leaner parameterisation from
# your preferred v2 (smaller cat‑emb floors, scaled numeric mixers, bias terms).
#
#   1.  **Exactly one 256‑d token per high‑level gameplay concept**
#         (GAME_STATE, SELF_INPUT, SELF_STATE, OPP_INPUT, OPP_STATE,
#          NANA_SELF, NANA_OPP, PROJECTILES)
#   2.  Each token is produced by a *CompositeToken* mini‑encoder that fuses
#       categorical IDs, floats and bools into a fixed‑width vector.
#       – Category fields → tiny learned embeddings (4 ≤ d ≤ 16)
#       – Float / Bool groups → skinny Linear → GELU (bias=True for flexibility)
#   3.  Two layers of self‑attention across the 8 tokens *within a frame* let
#       concepts exchange information.  A learnable [CLS] query pools them.
#   4.  The pooled 256‑d summary vector is LayerNorm + Linear‑projected up to
#       `d_model` (default 1024) so the *temporal* Transformer that follows sees
#       a familiar model width.
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import List, Mapping, Sequence

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Hyper‑params / helpers
# -----------------------------------------------------------------------------
DROPOUT_P: float = 0.10          # global dropout applied everywhere
D_INTRA:   int   = 256           # width of every concept token (post‑projection)


def _embed_dim(cardinality: int) -> int:
    """Heuristic: embed dims grow gently with |vocab|^0.25 (min 4).

    ‑ Very small enums (≤ 4 distinct values) get a 4‑d embedding.
    ‑ Mid‑sized enums (e.g. 90 characters) land around 12‑16 dims.
    """
    return max(4, int(cardinality ** 0.25 * 4))


# -----------------------------------------------------------------------------
#  CompositeToken – glue cats / floats / bools for *one* concept
# -----------------------------------------------------------------------------
class CompositeToken(nn.Module):
    """Fuses heterogeneous feature groups into a fixed‑width concept vector.

    Parameters
    ----------
    cat_specs : list[int]
        Cardinalities for *each* categorical field in the concept.
    n_float   : int
        Count of raw float features (already z‑scored by the Dataset collate).
    n_bool    : int
        Count of boolean indicator features (0/1, will be cast to float).
    d_out     : int
        Token width after projection (defaults to **D_INTRA = 256**).
    """

    def __init__(self, *, cat_specs: Sequence[int], n_float: int,
                 n_bool: int, d_out: int = D_INTRA) -> None:
        super().__init__()

        # ––– tiny embeddings per categorical field –––
        self.cat_embs = nn.ModuleList(
            nn.Embedding(card, _embed_dim(card)) for card in cat_specs)

        # ––– lightweight linears for numeric groups –––
        # Dim scales with feature count to avoid the fixed 64/32‑d cost in v1.
        float_dim = max(16, n_float * 4) if n_float else 0
        bool_dim  = max( 8, n_bool  * 4) if n_bool  else 0

        self.float_lin: nn.Module | None = (
            nn.Linear(n_float, float_dim, bias=True) if n_float else None)
        self.bool_lin:  nn.Module | None = (
            nn.Linear(n_bool,  bool_dim,  bias=True) if n_bool  else None)

        concat_dim = sum(e.embedding_dim for e in self.cat_embs) + float_dim + bool_dim

        # ––– final projection to token width –––
        self.mix = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, d_out, bias=True),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
        )

    # ---------------------------------------------------------------------
    def forward(self, *, cats: List[torch.Tensor],
                floats: torch.Tensor | None,
                bools: torch.Tensor | None) -> torch.Tensor:
        """All inputs arrive batched as (B, T, …) from the Dataset."""
        parts: List[torch.Tensor] = [emb(c) for emb, c in zip(self.cat_embs, cats)]
        if floats is not None:
            parts.append(self.float_lin(floats))       # type: ignore[arg-type]
        if bools is not None:
            parts.append(self.bool_lin(bools.float()))  # type: ignore[arg-type]
        return self.mix(torch.cat(parts, dim=-1))      # (B,T,256)


# -----------------------------------------------------------------------------
# Intra‑frame pooling via a *mini* Transformer (a.k.a. PMA / Set‑Attention)
# -----------------------------------------------------------------------------
class _GroupAttention(nn.Module):
    """Two‑layer self‑attention across the N concept tokens inside **one** frame."""

    def __init__(self, d_intra: int = D_INTRA, nhead: int = 4,
                 nlayers: int = 2, k_query: int = 1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_intra,
            nhead=nhead,
            dim_feedforward=4 * d_intra,
            dropout=DROPOUT_P,
            batch_first=True,   # expect (B, N, C)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)

        # Learnable [CLS] (or k>1) query tokens -----------------
        self.k_query = k_query
        self.queries = nn.Parameter(torch.randn(k_query, d_intra) * 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B*T, N, d_intra) → pooled (B*T, d_intra)."""
        bt = tokens.size(0)
        q = self.queries.unsqueeze(0).expand(bt, -1, -1)   # (B*T,k,d)
        h = self.encoder(torch.cat([q, tokens], dim=1))    # (B*T,k+N,d)
        return h[:, :self.k_query].mean(dim=1)             # average if k>1


# -----------------------------------------------------------------------------
# FrameEncoder – orchestrates token build ➜ intra‑frame attention ➜ projection
# -----------------------------------------------------------------------------
class FrameEncoder(nn.Module):
    """Encode a time‑sequence of structured frame dicts → (B, T, d_model)."""

    # ––– constants for Dataset numeric buckets –––
    STICK_FLOATS   = 8     # main_x, main_y, l, r ×2 (self / opp)
    BUTTON_BOOLS   = 12    # per player

    def __init__(self,
                 *,
                 num_stages: int,
                 num_characters: int,
                 num_actions: int,
                 num_costumes: int,
                 num_proj_types: int,
                 num_proj_subtypes: int,
                 d_model: int = 1024,
                 num_ports: int = 4,
                 proj_slots: int = 8) -> None:
        super().__init__()
        self.proj_slots = proj_slots

        # -----------------------------------------------------------------
        # 1.  Build one CompositeToken *module* per gameplay concept
        # -----------------------------------------------------------------
        self.tokens = nn.ModuleDict({
            # GAME_STATE  (stage, randall) + 14 blast‑zone/platform edges + distance
            "GAME_STATE": CompositeToken(cat_specs=[num_stages, 4],
                                         n_float=15, n_bool=0),
            # SELF / OPP inputs (analog + buttons)
            "SELF_INPUT": CompositeToken(cat_specs=[], n_float=8, n_bool=12),
            "OPP_INPUT" : CompositeToken(cat_specs=[], n_float=8, n_bool=12),
            # SELF / OPP state
            "SELF_STATE": CompositeToken(cat_specs=[num_ports, num_characters,
                                                     num_actions, num_costumes, 2],
                                         n_float=23, n_bool=9),
            "OPP_STATE" : CompositeToken(cat_specs=[num_ports, num_characters,
                                                     num_actions, num_costumes, 2],
                                         n_float=23, n_bool=9),
            # Nana tokens
            "NANA_SELF" : CompositeToken(cat_specs=[num_characters, num_actions],
                                         n_float=24, n_bool=17),
            "NANA_OPP"  : CompositeToken(cat_specs=[num_characters, num_actions],
                                         n_float=24, n_bool=17),
            # PROJECTILES – flatten *proj_slots* heterogeneous slots into one bundle
            "PROJECTILES": CompositeToken(cat_specs=[num_proj_types, num_proj_subtypes, 3]
                                               * proj_slots,
                                            n_float=5 * proj_slots,
                                            n_bool=0),
        })

        # -----------------------------------------------------------------
        # 2.  Intra‑frame cross‑attention & pooling
        # -----------------------------------------------------------------
        self.set_attn = _GroupAttention(d_intra=D_INTRA, nhead=4, nlayers=2)

        # -----------------------------------------------------------------
        # 3.  Projection to temporal Transformer width
        # -----------------------------------------------------------------
        self.out_proj = nn.Sequential(
            nn.LayerNorm(D_INTRA),
            nn.Linear(D_INTRA, d_model, bias=True),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
        )

    # ---------------------------------------------------------------------
    @staticmethod
    def _collapse(x: torch.Tensor) -> torch.Tensor:
        """(B, T, …) → (B*T, …) – handy for feeding *per‑frame* attention."""
        b, t = x.shape[:2]
        return x.reshape(b * t, *x.shape[2:])

    # ---------------------------------------------------------------------
    def forward(self, seq: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Build concept tokens, pool inside each frame, project to d_model.

        Notes
        -----
        All tensors in *seq* already have shape (B, T, …) and floats are
        standardised (mean 0, std 1) by the Dataset’s `collate` method.
        """
        toks: List[torch.Tensor] = []
        a = toks.append  # terse alias

        # ––– GAME_STATE –––
        blz = torch.stack([
            seq[k] for k in (
                "blastzone_left", "blastzone_right",
                "blastzone_top", "blastzone_bottom",
                "stage_edge_left", "stage_edge_right",
                "left_platform_height",  "left_platform_left",  "left_platform_right",
                "right_platform_height", "right_platform_left", "right_platform_right",
                "top_platform_height",   "top_platform_left",   "top_platform_right",
            )
        ], dim=-1)                                         # (B,T,14)

        a(self.tokens["GAME_STATE"](
            cats=[seq["stage"], seq["randall_state"]],
            floats=torch.cat([blz, seq["distance"].unsqueeze(-1)], dim=-1),
            bools=None))

        # ––– SELF / OPP INPUT TOKENS –––
        a(self.tokens["SELF_INPUT"](
            cats=[],
            floats=seq["self_sticks"],
            bools=seq["self_buttons"],
        ))
        a(self.tokens["OPP_INPUT"](
            cats=[],
            floats=seq["opp_sticks"],
            bools=seq["opp_buttons"],
        ))

        # ––– SELF / OPP STATE TOKENS –––
        a(self.tokens["SELF_STATE"](
            cats=[seq["self_port"], seq["self_character"],
                  seq["self_action"], seq["self_costume"], seq["self_facing"]],
            floats=seq["self_state_floats"],
            bools=seq["self_state_flags"],
        ))
        a(self.tokens["OPP_STATE"](
            cats=[seq["opp_port"], seq["opp_character"],
                  seq["opp_action"], seq["opp_costume"], seq["opp_facing"]],
            floats=seq["opp_state_floats"],
            bools=seq["opp_state_flags"],
        ))

        # ––– NANA TOKENS –––
        a(self.tokens["NANA_SELF"](
            cats=[seq["self_nana_character"], seq["self_nana_action"]],
            floats=seq["self_nana_floats"],
            bools=seq["self_nana_flags"],
        ))
        a(self.tokens["NANA_OPP"](
            cats=[seq["opp_nana_character"], seq["opp_nana_action"]],
            floats=seq["opp_nana_floats"],
            bools=seq["opp_nana_flags"],
        ))

        # ––– PROJECTILES (flatten N slots) –––
        proj_cats: List[torch.Tensor] = []
        proj_floats: List[torch.Tensor] = []
        for j in range(self.proj_slots):
            proj_cats.extend([
                seq[f"proj{j}_type"],
                seq[f"proj{j}_subtype"],
                seq[f"proj{j}_owner"],
            ])
            proj_floats.append(seq[f"proj{j}_floats"])       # (B,T,5)

        a(self.tokens["PROJECTILES"](
            cats=proj_cats,
            floats=torch.cat(proj_floats, dim=-1),           # (B,T,5*slots)
            bools=None,
        ))

        # -----------------------------------------------------------------
        # 2.  Stack (B,T,N,D) → pool via intra‑frame attention
        # -----------------------------------------------------------------
        group = torch.stack(toks, dim=2)                    # (B,T,8,256)
        pooled = self.set_attn(self._collapse(group))       # (B*T,256)
        pooled = pooled.view(group.size(0), group.size(1), D_INTRA)

        # -----------------------------------------------------------------
        # 3.  Project to temporal Transformer width
        # -----------------------------------------------------------------
        return self.out_proj(pooled)                        # (B,T,d_model)
