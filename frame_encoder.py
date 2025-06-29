# frame_encoder.py – intra-frame cross-attention (Composite-Token v2, spec-aligned)
# -----------------------------------------------------------------------------
# Encodes one structured Melee frame → (B, T, d_model)
#
# Eight 256-d “concept tokens” per frame:
#   GAME_STATE, SELF_INPUT, SELF_STATE, OPP_INPUT, OPP_STATE,
#   NANA_SELF, NANA_OPP, PROJECTILES
#
# Authoritative counts (2025-06):
#   • GAME_STATE: 1 cat, 18 floats, 0 bool
#   • SELF/OPP INPUT: 1 cat, 8 floats, 12 bool
#   • SELF/OPP STATE: 4 cats, 23 floats, 5 bool
#   • NANA SELF/OPP: 3 cats, 24 floats, 5 bool
#   • PROJECTILES (8 slots): 3 cats/slot, 5 floats/slot, 0 bool
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import List, Mapping, Sequence
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────
DROPOUT_P: float = 0.10
D_INTRA:   int   = 256


def _embed_dim(cardinality: int) -> int:
    return max(4, int(cardinality ** 0.25 * 4))


# ──────────────────────────────────────────────────────────────────────
#  CompositeToken
# ──────────────────────────────────────────────────────────────────────
class CompositeToken(nn.Module):
    def __init__(
        self,
        *,
        cat_specs: Sequence[int],
        n_float: int,
        n_bool: int,
        d_out: int = D_INTRA,
    ) -> None:
        super().__init__()

        self.cat_embs = nn.ModuleList(
            nn.Embedding(card, _embed_dim(card)) for card in cat_specs
        )

        float_dim = max(16, n_float * 4) if n_float else 0
        bool_dim  = max( 8, n_bool  * 4) if n_bool  else 0

        self.float_lin = nn.Linear(n_float, float_dim, bias=True) if n_float else None
        self.bool_lin  = nn.Linear(n_bool,  bool_dim,  bias=True) if n_bool  else None

        concat_dim = sum(e.embedding_dim for e in self.cat_embs) + float_dim + bool_dim
        self.mix = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, d_out, bias=True),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
        )

    def forward(
        self,
        *,
        cats: List[torch.Tensor],
        floats: torch.Tensor | None,
        bools: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = [emb(c) for emb, c in zip(self.cat_embs, cats)]
        if floats is not None:
            parts.append(self.float_lin(floats))        # type: ignore[arg-type]
        if bools is not None:
            parts.append(self.bool_lin(bools.float()))  # type: ignore[arg-type]
        return self.mix(torch.cat(parts, dim=-1))       # (B,T,256)


# ──────────────────────────────────────────────────────────────────────
#  Tiny Transformer for intra-frame pooling
# ──────────────────────────────────────────────────────────────────────
class _GroupAttention(nn.Module):
    def __init__(self, d_intra: int = D_INTRA, nhead: int = 4,
                 nlayers: int = 2, k_query: int = 1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_intra,
            nhead=nhead,
            dim_feedforward=4 * d_intra,
            dropout=DROPOUT_P,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.k_query = k_query
        self.queries = nn.Parameter(torch.randn(k_query, d_intra) * 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bt = tokens.size(0)
        q  = self.queries.unsqueeze(0).expand(bt, -1, -1)
        h  = self.encoder(torch.cat([q, tokens], dim=1))
        return h[:, :self.k_query].mean(dim=1)          # (B*T,256)


# ──────────────────────────────────────────────────────────────────────
#  FrameEncoder
# ──────────────────────────────────────────────────────────────────────
class FrameEncoder(nn.Module):

    STICK_FLOATS = 8   # self + opp sticks
    BTN_BOOLS    = 12  # per player

    def __init__(
        self,
        *,
        num_stages: int,
        num_characters: int,
        num_actions: int,
        num_costumes: int,
        num_proj_types: int,
        num_proj_subtypes: int,
        num_c_dirs: int,
        d_model: int = 1024,
        num_ports: int = 4,
        proj_slots: int = 8,
    ) -> None:
        super().__init__()
        self.proj_slots = proj_slots

        # 1. Composite tokens ------------------------------------------------
        self.tokens = nn.ModuleDict({
            "GAME_STATE": CompositeToken(
                cat_specs=[num_stages],        # stage only
                n_float=20, n_bool=0
            ),
            "SELF_INPUT": CompositeToken(
                cat_specs=[num_c_dirs],
                n_float=self.STICK_FLOATS, n_bool=self.BTN_BOOLS
            ),
            "OPP_INPUT": CompositeToken(
                cat_specs=[num_c_dirs],
                n_float=self.STICK_FLOATS, n_bool=self.BTN_BOOLS
            ),
            "SELF_STATE": CompositeToken(
                cat_specs=[num_ports, num_characters, num_actions, num_costumes],
                n_float=23, n_bool=5
            ),
            "OPP_STATE": CompositeToken(
                cat_specs=[num_ports, num_characters, num_actions, num_costumes],
                n_float=23, n_bool=5
            ),
            "NANA_SELF": CompositeToken(
                cat_specs=[num_characters, num_actions, num_c_dirs],
                n_float=24, n_bool=5
            ),
            "NANA_OPP": CompositeToken(
                cat_specs=[num_characters, num_actions, num_c_dirs],
                n_float=24, n_bool=5
            ),
            "PROJECTILES": CompositeToken(
                cat_specs=[num_proj_types, num_proj_subtypes, 3] * proj_slots,
                n_float=5 * proj_slots,
                n_bool=0
            ),
        })

        # 2. Intra-frame attention ------------------------------------------
        self.set_attn = _GroupAttention(D_INTRA, nhead=4, nlayers=2)

        # 3. Projection to d_model ------------------------------------------
        self.out_proj = nn.Sequential(
            nn.LayerNorm(D_INTRA),
            nn.Linear(D_INTRA, d_model, bias=True),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
        )

    # ---------------------------------------------------------------------
    @staticmethod
    def _collapse(x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:])

    # ---------------------------------------------------------------------
    def forward(self, seq: Mapping[str, torch.Tensor]) -> torch.Tensor:
        toks: List[torch.Tensor] = []
        a = toks.append

        #── GAME_STATE ──────────────────────────────────────────────────
        geom = torch.stack([
            seq[k] for k in (
                "blastzone_left",  "blastzone_right",
                "blastzone_top",   "blastzone_bottom",
                "stage_edge_left", "stage_edge_right",
                "left_platform_height",  "left_platform_left",  "left_platform_right",
                "right_platform_height", "right_platform_left", "right_platform_right",
                "top_platform_height",   "top_platform_left",   "top_platform_right",
            )
        ], dim=-1)                                     # (B,T,14)

        extras = torch.stack([
            seq["randall_height"],
            seq["randall_left"],
            seq["randall_right"],
        ], dim=-1)                                     # (B,T,3)

        floats_gs = torch.cat(
            [geom, extras, seq["distance"].unsqueeze(-1), seq["frame"].unsqueeze(-1)], dim=-1
        )                                               # (B,T,20)

        a(self.tokens["GAME_STATE"](
            cats=[seq["stage"]],
            floats=floats_gs,
            bools=None,
        ))

        #── INPUT TOKENS ───────────────────────────────────────────────
        a(self.tokens["SELF_INPUT"](
            cats=[seq["self_c_dir"]],
            floats=seq["self_sticks"],
            bools=seq["self_buttons"],
        ))
        a(self.tokens["OPP_INPUT"](
            cats=[seq["opp_c_dir"]],
            floats=seq["opp_sticks"],
            bools=seq["opp_buttons"],
        ))

        #── PLAYER STATE TOKENS ────────────────────────────────────────
        a(self.tokens["SELF_STATE"](
            cats=[seq["self_port"], seq["self_character"],
                  seq["self_action"], seq["self_costume"]],
            floats=seq["self_state_floats"],
            bools=seq["self_state_flags"],
        ))
        a(self.tokens["OPP_STATE"](
            cats=[seq["opp_port"], seq["opp_character"],
                  seq["opp_action"], seq["opp_costume"]],
            floats=seq["opp_state_floats"],
            bools=seq["opp_state_flags"],
        ))

        #── NANA TOKENS ────────────────────────────────────────────────
        a(self.tokens["NANA_SELF"](
            cats=[seq["self_nana_character"], seq["self_nana_action"],
                  seq["self_nana_c_dir"]],
            floats=seq["self_nana_floats"],
            bools=seq["self_nana_flags"],
        ))
        a(self.tokens["NANA_OPP"](
            cats=[seq["opp_nana_character"], seq["opp_nana_action"],
                  seq["opp_nana_c_dir"]],
            floats=seq["opp_nana_floats"],
            bools=seq["opp_nana_flags"],
        ))

        #── PROJECTILES ────────────────────────────────────────────────
        proj_cats, proj_floats = [], []
        for j in range(self.proj_slots):
            proj_cats.extend([
                seq[f"proj{j}_type"],
                seq[f"proj{j}_subtype"],
                seq[f"proj{j}_owner"],
            ])
            proj_floats.append(seq[f"proj{j}_floats"])  # (B,T,5)

        a(self.tokens["PROJECTILES"](
            cats=proj_cats,
            floats=torch.cat(proj_floats, dim=-1),      # (B,T,40)
            bools=None,
        ))

        #── Pool inside frame & project ────────────────────────────────
        group  = torch.stack(toks, dim=2)               # (B,T,8,256)
        pooled = self.set_attn(self._collapse(group))   # (B*T,256)
        pooled = pooled.view(group.size(0), group.size(1), D_INTRA)
        return self.out_proj(pooled)                    # (B,T,d_model)
