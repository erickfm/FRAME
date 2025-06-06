from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

# ----------------------------------------------------------------------------
# Helper generators ­– keep all column‑name logic in one place
# ----------------------------------------------------------------------------
BTN = [
    "BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
    "BUTTON_Z", "BUTTON_L", "BUTTON_R", "BUTTON_START",
    "BUTTON_D_UP", "BUTTON_D_DOWN", "BUTTON_D_LEFT", "BUTTON_D_RIGHT",
]


def btn_cols(prefix: str) -> List[str]:
    return [f"{prefix}_btn_{b}" for b in BTN]


def analog_cols(prefix: str) -> List[str]:
    return [
        f"{prefix}_main_x", f"{prefix}_main_y",
        f"{prefix}_c_x", f"{prefix}_c_y",
        f"{prefix}_l_shldr", f"{prefix}_r_shldr",
    ]


def numeric_state(prefix: str) -> List[str]:
    """Full set of continuous stats for a (player|nana) entity, incl. ECB."""
    base = [
        f"{prefix}_pos_x", f"{prefix}_pos_y",
        f"{prefix}_percent", f"{prefix}_stock",
        f"{prefix}_jumps_left",
        f"{prefix}_speed_air_x_self", f"{prefix}_speed_ground_x_self",
        f"{prefix}_speed_x_attack", f"{prefix}_speed_y_attack",
        f"{prefix}_speed_y_self",
        f"{prefix}_hitlag_left", f"{prefix}_hitstun_left",
        f"{prefix}_invuln_left", f"{prefix}_shield_strength",
    ]
    ecb = [
        f"{prefix}_ecb_{part}_{axis}"
        for part in ("bottom", "left", "right", "top")
        for axis in ("x", "y")
    ]
    return base + ecb


def flags(prefix: str) -> List[str]:
    return [
        f"{prefix}_on_ground", f"{prefix}_off_stage",
        f"{prefix}_facing", f"{prefix}_invulnerable",
        f"{prefix}_moonwalkwarning",
    ]


def categorical_ids(prefix: str) -> List[str]:
    return [f"{prefix}_port", f"{prefix}_character", f"{prefix}_action", f"{prefix}_costume"]


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class MeleeFrameDatasetWithDelay(Dataset):
    """Fixed‑length windows over Slippi frame data with reaction delay."""

    # Single‑frame, stage‑level geometry – identical for every entity
    _stage_geom_cols = [
        "blastzone_left", "blastzone_right", "blastzone_top", "blastzone_bottom",
        "stage_edge_left", "stage_edge_right",
        "left_platform_height", "left_platform_left", "left_platform_right",
        "right_platform_height", "right_platform_left", "right_platform_right",
        "top_platform_height", "top_platform_left", "top_platform_right",
        "randall_height", "randall_left", "randall_right",
    ]

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        parquet_dir: str,
        sequence_length: int = 30,
        reaction_delay: int = 6,
    ) -> None:
        super().__init__()

        self.parquet_dir = Path(parquet_dir)
        self.sequence_length = sequence_length
        self.reaction_delay = reaction_delay

        # 1) discover files -------------------------------------------------
        self.files = sorted(self.parquet_dir.glob("*.parquet"))
        if not self.files:
            raise RuntimeError(f"No .parquet files found in {parquet_dir}")

        # 2) map every valid window across all files -----------------------
        self.index_map: List[Tuple[Path, int]] = []
        for f in self.files:
            df = pd.read_parquet(f)
            df = df[df["frame"] >= 0]
            max_start = len(df) - (sequence_length + reaction_delay)
            if max_start > 0:
                self.index_map.extend([(f, s) for s in range(max_start)])
        if not self.index_map:
            raise RuntimeError("No valid windows across the dataset.")

        # 3) hierarchical feature spec -------------------------------------
        self.feature_groups: Dict[str, Dict] = self._build_feature_groups()

        # 4) collect categorical columns -----------------------------------
        self._categorical_cols: List[str] = []
        for _, meta in self._walk_groups(return_meta=True):
            if meta["ftype"] == "categorical":
                self._categorical_cols.extend(meta["cols"])

        # 5) build raw→index mappings for categorical features -------------
        self._build_categorical_mappings()

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    def _build_feature_groups(self) -> Dict[str, Dict]:
        """Return the nested dict describing all feature groups."""
        fg: Dict[str, Dict] = {
            "global": {
                "numeric": ["distance", "frame", *self._stage_geom_cols],
                "categorical": ["stage"],
            },
            "players": {
                "self": {
                    "categorical": categorical_ids("self"),
                    "buttons": btn_cols("self"),
                    "flags": flags("self"),
                    "analog": analog_cols("self"),
                    "numeric": numeric_state("self"),
                    "action_elapsed": ["self_action_frame"],
                },
                "opp": {
                    "categorical": categorical_ids("opp"),
                    "buttons": btn_cols("opp"),
                    "flags": flags("opp"),
                    "analog": analog_cols("opp"),
                    "numeric": numeric_state("opp"),
                    "action_elapsed": ["opp_action_frame"],
                },
                "self_nana": {
                    "categorical": ["self_nana_character", "self_nana_action"],
                    "buttons": btn_cols("self_nana"),
                    "flags": flags("self_nana") + ["self_nana_present"],
                    "analog": analog_cols("self_nana"),
                    "numeric": numeric_state("self_nana") + [
                        "self_nana_stock", "self_nana_jumps_left",
                        "self_nana_hitlag_left", "self_nana_hitstun_left",
                        "self_nana_invuln_left",
                    ],
                    "action_elapsed": ["self_nana_action_frame"],
                },
                "opp_nana": {
                    "categorical": ["opp_nana_character", "opp_nana_action"],
                    "buttons": btn_cols("opp_nana"),
                    "flags": flags("opp_nana") + ["opp_nana_present"],
                    "analog": analog_cols("opp_nana"),
                    "numeric": numeric_state("opp_nana") + [
                        "opp_nana_stock", "opp_nana_jumps_left",
                        "opp_nana_hitlag_left", "opp_nana_hitstun_left",
                        "opp_nana_invuln_left",
                    ],
                    "action_elapsed": ["opp_nana_action_frame"],
                },
            },
            "projectiles": {
                k: {
                    "categorical": [f"proj{k}_owner", f"proj{k}_type", f"proj{k}_subtype"],
                    "numeric": [
                        f"proj{k}_pos_x", f"proj{k}_pos_y",
                        f"proj{k}_speed_x", f"proj{k}_speed_y",
                        f"proj{k}_frame",
                    ],
                }
                for k in range(8)
            },
        }
        return fg

    def _walk_groups(self, return_meta=False):
        """Yield (key_chain, meta) for every leaf feature list inside fg."""
        stack = [((), self.feature_groups)]
        while stack:
            prefix, node = stack.pop()
            if isinstance(node, dict) and all(k in (
                "numeric", "categorical", "buttons", "flags", "analog", "action_elapsed"
            ) for k in node):
                # leaf mapping
                for ftype, cols in node.items():
                    if cols:
                        meta = {
                            "ftype": ftype,
                            "cols": cols,
                            "entity": prefix[-1] if prefix else "global",
                        }
                        if return_meta:
                            yield prefix, meta
                        else:
                            yield cols
            else:
                for k, sub in node.items():
                    stack.append(((*prefix, k), sub))

    def _build_categorical_mappings(self):
        """Create raw‑id → contiguous‑id maps per categorical column."""
        raw_unique = {c: set() for c in self._categorical_cols}
        for f in self.files:
            df = pd.read_parquet(f)
            df = df[df["frame"] >= 0]
            for c in self._categorical_cols:
                if c in df.columns:
                    vals = df[c].dropna().astype("int64")
                    raw_unique[c].update(vals[vals >= 0].tolist())
        for c, s in raw_unique.items():
            vals = sorted(s)
            if 0 not in vals:
                vals.insert(0, 0)
            setattr(self, f"{c}_map", {raw: idx for idx, raw in enumerate(vals)})

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        fpath, start_idx = self.index_map[idx]
        W, R = self.sequence_length, self.reaction_delay

        df = pd.read_parquet(fpath)
        df = df[df["frame"] >= 0].reset_index(drop=True).fillna(0.0)

        # remap categorical → small int
        for c in self._categorical_cols:
            raw_map = getattr(self, f"{c}_map")
            df[c] = df[c].clip(lower=0).map(raw_map).fillna(0).astype("int64")

        # synthetic Nana‑presence flags (used by downstream model)
        df["self_nana_present"] = (df["self_nana_character"] >= 0).astype("float32")
        df["opp_nana_present"] = (df["opp_nana_character"] >= 0).astype("float32")

        end_idx = start_idx + W
        target_idx = end_idx + R - 1

        slice_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        target_row = df.iloc[target_idx]

        state_seq: Dict[str, torch.Tensor] = {}

        # iterate hierarchical spec --------------------------------------
        for _, meta in self._walk_groups(return_meta=True):
            cols, ftype = meta["cols"], meta["ftype"]
            key = f"{meta['entity']}_{ftype}" if meta['entity'] != "global" else ftype

            if ftype == "categorical":
                for c in cols:
                    state_seq[c] = torch.from_numpy(slice_df[c].values).long()
            else:  # numeric / analog / buttons / flags / action_elapsed
                arrs = [torch.from_numpy(slice_df[c].astype("float32").values) for c in cols]
                state_seq[key] = torch.stack(arrs, dim=-1) if len(arrs) > 1 else arrs[0]

        # --------------------------------------------------------------
        # target – future self inputs (same as original implementation)
        # --------------------------------------------------------------
        target = {
            "main_x": torch.tensor(target_row["self_main_x"], dtype=torch.float32),
            "main_y": torch.tensor(target_row["self_main_y"], dtype=torch.float32),
            "c_x": torch.tensor(target_row["self_c_x"], dtype=torch.float32),
            "c_y": torch.tensor(target_row["self_c_y"], dtype=torch.float32),
            "l_shldr": torch.tensor(target_row["self_l_shldr"], dtype=torch.float32),
            "r_shldr": torch.tensor(target_row["self_r_shldr"], dtype=torch.float32),
        }
        target["btns"] = torch.stack(
            [torch.tensor(target_row[c], dtype=torch.float32) for c in btn_cols("self")],
            dim=0,
        )

        return state_seq, target
