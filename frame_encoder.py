import torch
import torch.nn as nn
from typing import Dict

# Global dropout probability applied consistently
DROPOUT_P = 0.10


# ────────────────────────────────────────────────────────────────
# Helper: MLP block with LayerNorm → Linear → ReLU → Dropout
# ────────────────────────────────────────────────────────────────
def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Dropout(DROPOUT_P),
    )


# ────────────────────────────────────────────────────────────────
# FrameEncoder
# ────────────────────────────────────────────────────────────────
class FrameEncoder(nn.Module):
    """
    High-capacity encoder that maps structured Melee frame data
    to a dense (B, T, 256) embedding, with per-feature dims and dropout.
    """

    # Raw input dimensions from dataset spec
    GLOBAL_NUM   = 20
    PLAYER_NUM   = 22
    NANA_NUM     = 27
    ANALOG_DIM   = 6
    BTN_DIM      = 12
    FLAGS_DIM    = 5
    NANA_FLAGS   = 6
    PROJ_NUM_PER = 5
    PROJ_SLOTS   = 8

    # Per-group output dimensions
    FEATURE_DIMS = {
        # Categorical groups
        "stage":        64,
        "port":         32,
        "character":    128,
        "action":       128,
        "costume":      1,
        "proj_owner":   32,
        "proj_type":    64,
        "proj_subtype": 64,

        # Float / numeric groups
        "global_numeric":   64,
        "player_numeric":   128,
        "nana_numeric":     96,
        "analog":           64,
        "proj_numeric":     64,

        # Boolean groups
        "buttons":      64,
        "flags":        64,
        "nana_buttons": 64,
        "nana_flags":   64,
    }

    def __init__(
        self,
        num_stages: int,
        num_ports: int,
        num_characters: int,
        num_actions: int,
        num_costumes: int,
        num_proj_types: int,
        num_proj_subtypes: int,
        d_model: int = 256,
    ):
        super().__init__()
        D = self.FEATURE_DIMS  # shorthand

        # Categorical embeddings
        self.stage_emb  = nn.Embedding(num_stages,   D["stage"])
        self.port_emb   = nn.Embedding(num_ports,    D["port"])
        self.char_emb   = nn.Embedding(num_characters, D["character"])
        self.act_emb    = nn.Embedding(num_actions,  D["action"])
        self.cost_emb   = nn.Embedding(num_costumes, D["costume"])
        self.ptype_emb  = nn.Embedding(num_proj_types,   D["proj_type"])
        self.psub_emb   = nn.Embedding(num_proj_subtypes, D["proj_subtype"])

        # Float / boolean encoders (include action_elapsed)
        self.glob_enc       = _mlp(self.GLOBAL_NUM,                 D["global_numeric"])
        self.num_enc        = _mlp(self.PLAYER_NUM + 1,            D["player_numeric"])
        self.nana_num_enc   = _mlp(self.NANA_NUM + 1,              D["nana_numeric"])
        self.analog_enc     = _mlp(self.ANALOG_DIM,                 D["analog"])
        self.proj_num_enc   = _mlp(self.PROJ_NUM_PER * self.PROJ_SLOTS, D["proj_numeric"])
        self.btn_enc        = _mlp(self.BTN_DIM * 2,                D["buttons"])
        self.flag_enc       = _mlp(self.FLAGS_DIM * 2,              D["flags"])
        self.nana_btn_enc   = _mlp(self.BTN_DIM * 2,                D["nana_buttons"])
        self.nana_flag_enc  = _mlp(self.NANA_FLAGS * 2,             D["nana_flags"])

        # Compute total concat dimension automatically
        self.total_dim = sum([
            D["stage"],
            2 * D["port"],
            4 * D["character"],
            4 * D["action"],
            2 * D["costume"],
            self.PROJ_SLOTS * (D["proj_owner"] + D["proj_type"] + D["proj_subtype"]),
            D["global_numeric"],
            2 * D["player_numeric"],
            2 * D["nana_numeric"],
            4 * D["analog"],
            D["proj_numeric"],
            D["buttons"],
            D["flags"],
            D["nana_buttons"],
            D["nana_flags"],
        ])

        # Dropout applied to concatenated features
        self.concat_dropout = nn.Dropout(DROPOUT_P)

        # Final projection to d_model
        self.proj = nn.Sequential(
            nn.LayerNorm(self.total_dim),
            nn.Linear(self.total_dim, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT_P),
        )

    def _embed(self, emb: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        return emb(x.flatten()).reshape(B, T, -1)

    def _apply_mlp(self, mlp: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        return mlp(x.reshape(B * T, -1)).reshape(B, T, -1)

    def forward(self, seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        seq : Mapping[str, Tensor]
            All tensors already shaped (B, T, …).

        Returns
        -------
        Tensor
            (B, T, 256) frame embeddings.
        """
        # ── Categorical embeddings
        cat_parts = [
            self._embed(self.stage_emb,   seq["stage"]),
            self._embed(self.port_emb,    seq["self_port"]),
            self._embed(self.port_emb,    seq["opp_port"]),
            self._embed(self.char_emb,    seq["self_character"]),
            self._embed(self.char_emb,    seq["opp_character"]),
            self._embed(self.act_emb,     seq["self_action"]),
            self._embed(self.act_emb,     seq["opp_action"]),
            self._embed(self.cost_emb,    seq["self_costume"]),
            self._embed(self.cost_emb,    seq["opp_costume"]),
            self._embed(self.char_emb,    seq["self_nana_character"]),
            self._embed(self.char_emb,    seq["opp_nana_character"]),
            self._embed(self.act_emb,     seq["self_nana_action"]),
            self._embed(self.act_emb,     seq["opp_nana_action"]),
        ]
        # Projectile categorical embeddings
        for j in range(self.PROJ_SLOTS):
            cat_parts.append(self._embed(self.port_emb,  seq[f"proj{j}_owner"]))
            cat_parts.append(self._embed(self.ptype_emb, seq[f"proj{j}_type"]))
            cat_parts.append(self._embed(self.psub_emb,  seq[f"proj{j}_subtype"]))

        # ── Numeric / boolean encodings (action_elapsed folded into numeric)
        dense_parts = [
            self._apply_mlp(self.glob_enc,       seq["numeric"]),
            self._apply_mlp(
                self.num_enc,
                torch.cat([seq["self_numeric"], seq["self_action_elapsed"].unsqueeze(-1).float()], dim=-1)
            ),
            self._apply_mlp(
                self.num_enc,
                torch.cat([seq["opp_numeric"], seq["opp_action_elapsed"].unsqueeze(-1).float()], dim=-1)
            ),
            self._apply_mlp(
                self.nana_num_enc,
                torch.cat([seq["self_nana_numeric"], seq["self_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1)
            ),
            self._apply_mlp(
                self.nana_num_enc,
                torch.cat([seq["opp_nana_numeric"], seq["opp_nana_action_elapsed"].unsqueeze(-1).float()], dim=-1)
            ),
            self._apply_mlp(self.analog_enc,     seq["self_analog"]),
            self._apply_mlp(self.analog_enc,     seq["opp_analog"]),
            self._apply_mlp(self.analog_enc,     seq["self_nana_analog"]),
            self._apply_mlp(self.analog_enc,     seq["opp_nana_analog"]),
            self._apply_mlp(
                self.proj_num_enc,
                torch.cat([seq[f"{k}_numeric"] for k in map(str, range(self.PROJ_SLOTS))], dim=-1),
            ),
            self._apply_mlp(
                self.btn_enc,
                torch.cat([seq["self_buttons"].float(), seq["opp_buttons"].float()], dim=-1),
            ),
            self._apply_mlp(
                self.flag_enc,
                torch.cat([seq["self_flags"].float(), seq["opp_flags"].float()], dim=-1),
            ),
            self._apply_mlp(
                self.nana_btn_enc,
                torch.cat([seq["self_nana_buttons"].float(), seq["opp_nana_buttons"].float()], dim=-1),
            ),
            self._apply_mlp(
                self.nana_flag_enc,
                torch.cat([seq["self_nana_flags"].float(), seq["opp_nana_flags"].float()], dim=-1),
            ),
        ]

        # ── Concatenate all features, apply dropout, project
        concat = torch.cat(cat_parts + dense_parts, dim=-1)     # (B, T, total_dim)
        concat = self.concat_dropout(concat)                    # dropout on full vector
        return self.proj(concat)                                # (B, T, 256)
