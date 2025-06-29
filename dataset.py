# dataset.py
# -------------------------------------------------------------------------
# Slippi parquet → fixed-length windows for FRAME
# Ensures no NaNs or Infs in any numeric feature (including distance)
# -------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cat_maps import (
    STAGE_MAP,
    CHARACTER_MAP,
    ACTION_MAP,
    PROJECTILE_TYPE_MAP,
)

# -----------------------------------------------------------------------------
# Helper generators – keep all column-name logic in one place
# -----------------------------------------------------------------------------
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
        f"{prefix}_l_shldr", f"{prefix}_r_shldr",
    ]

def numeric_state(prefix: str) -> List[str]:
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
    return [
        f"{prefix}_port",
        f"{prefix}_character",
        f"{prefix}_action",
        f"{prefix}_costume",
    ]

# -----------------------------------------------------------------------------
# C-stick direction encoding
# -----------------------------------------------------------------------------
def encode_cstick_dir(
    df: pd.DataFrame,
    prefix: str,
    dead_zone: float = 0.15,
) -> str:
    dx = df[f"{prefix}_c_x"].astype("float32") - 0.5
    dy = df[f"{prefix}_c_y"].astype("float32") - 0.5

    mag   = np.hypot(dx, dy)
    cat   = np.zeros_like(mag, dtype="int64")
    alive = mag > dead_zone

    horiz = alive & (np.abs(dx) >= np.abs(dy))
    vert  = alive & (np.abs(dy) >  np.abs(dx))

    cat[horiz & (dx > 0)] = 4
    cat[horiz & (dx < 0)] = 3
    cat[vert  & (dy > 0)] = 1
    cat[vert  & (dy < 0)] = 2

    new_col = f"{prefix}_c_dir"
    df[new_col] = cat
    return new_col

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class MeleeFrameDatasetWithDelay(Dataset):
    """Fixed-length windows over Slippi frame data with a reaction delay."""

    _stage_geom_cols = [
        "blastzone_left", "blastzone_right", "blastzone_top", "blastzone_bottom",
        "stage_edge_left", "stage_edge_right",
        "left_platform_height",  "left_platform_left",  "left_platform_right",
        "right_platform_height", "right_platform_left", "right_platform_right",
        "top_platform_height",   "top_platform_left",   "top_platform_right",
        "randall_height", "randall_left", "randall_right",
    ]

    def __init__(
        self,
        parquet_dir: str,
        sequence_length: int = 30,
        reaction_delay: int = 1,
    ) -> None:
        super().__init__()
        self.parquet_dir     = Path(parquet_dir)
        self.sequence_length = sequence_length
        self.reaction_delay  = reaction_delay

        self.files = sorted(self.parquet_dir.glob("*.parquet"))
        if not self.files:
            raise RuntimeError(f"No .parquet files found in {parquet_dir}")

        # Map every valid window across all parquet files
        self.index_map: List[Tuple[Path, int]] = []
        for f in self.files:
            df = pd.read_parquet(f)
            df = df[df["frame"] >= 0]
            max_start = len(df) - (sequence_length + reaction_delay)
            if max_start > 0:
                self.index_map.extend([(f, s) for s in range(max_start)])
        if not self.index_map:
            raise RuntimeError("No valid windows across the dataset.")

        # Build feature spec & collect categorical columns
        self.feature_groups: Dict[str, Dict] = self._build_feature_groups()
        self._categorical_cols: List[str] = []
        for _, meta in self._walk_groups(return_meta=True):
            if meta["ftype"] == "categorical":
                self._categorical_cols.extend(meta["cols"])

        # Build dynamic categorical maps
        self._build_categorical_mappings()

        # Fixed enum maps
        self._enum_maps: Dict[str, Dict[int, int]] = {
            "stage": STAGE_MAP,
            "_character": CHARACTER_MAP,
            "_action":    ACTION_MAP,
            "_type":      PROJECTILE_TYPE_MAP,
            "c_dir":      {i: i for i in range(5)},
        }

    def _get_enum_map(self, col: str) -> Dict[int, int]:
        if col == "stage":
            return self._enum_maps["stage"]
        if col.endswith("_c_dir"):
            return self._enum_maps["c_dir"]
        for suffix, m in self._enum_maps.items():
            if suffix != "stage" and col.endswith(suffix):
                return m
        return getattr(self, f"{col}_map")

    def _build_feature_groups(self) -> Dict[str, Dict]:
        fg: Dict[str, Dict] = {
            "global": {
                "numeric": ["distance", "frame", *self._stage_geom_cols],
                "categorical": ["stage"],
            },
            "players": {
                "self": {
                    "categorical": categorical_ids("self") + ["self_c_dir"],
                    "buttons": btn_cols("self"),
                    "flags":   flags("self"),
                    "analog":  analog_cols("self"),
                    "numeric": numeric_state("self"),
                    "action_elapsed": ["self_action_frame"],
                },
                "opp": {
                    "categorical": categorical_ids("opp") + ["opp_c_dir"],
                    "buttons": btn_cols("opp"),
                    "flags":   flags("opp"),
                    "analog":  analog_cols("opp"),
                    "numeric": numeric_state("opp"),
                    "action_elapsed": ["opp_action_frame"],
                },
                "self_nana": {
                    "categorical": [
                        "self_nana_character", "self_nana_action", "self_nana_c_dir"
                    ],
                    "buttons": btn_cols("self_nana"),
                    "flags":   flags("self_nana") + ["self_nana_present"],
                    "analog":  analog_cols("self_nana"),
                    "numeric": numeric_state("self_nana") + [
                        "self_nana_stock", "self_nana_jumps_left",
                        "self_nana_hitlag_left", "self_nana_hitstun_left",
                        "self_nana_invuln_left",
                    ],
                    "action_elapsed": ["self_nana_action_frame"],
                },
                "opp_nana": {
                    "categorical": [
                        "opp_nana_character", "opp_nana_action", "opp_nana_c_dir"
                    ],
                    "buttons": btn_cols("opp_nana"),
                    "flags":   flags("opp_nana") + ["opp_nana_present"],
                    "analog":  analog_cols("opp_nana"),
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
                    "categorical": [
                        f"proj{k}_owner", f"proj{k}_type", f"proj{k}_subtype"
                    ],
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

    def _walk_groups(self, *, return_meta=False):
        stack = [((), self.feature_groups)]
        while stack:
            prefix, node = stack.pop()
            if isinstance(node, dict) and all(
                k in ("numeric", "categorical", "buttons", "flags", "analog", "action_elapsed")
                for k in node
            ):
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
        raw_unique = {
            c: set()
            for c in self._categorical_cols
            if c not in {"stage"}
            and not c.endswith("_character")
            and not c.endswith("_action")
            and not c.endswith("_type")
            and not c.endswith("_c_dir")
        }
        for fpath in self.files:
            df = pd.read_parquet(fpath)
            df = df[df["frame"] >= 0]
            for c in list(raw_unique):
                if c in df.columns:
                    vals = df[c].dropna().astype("int64")
                    raw_unique[c].update(v for v in vals if 0 <= v <= 4)
        for c, s in raw_unique.items():
            vals = sorted(s)
            if 0 not in vals:
                vals.insert(0, 0)
            setattr(self, f"{c}_map", {raw: idx for idx, raw in enumerate(vals)})

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        fpath, start_idx = self.index_map[idx]
        W, R = self.sequence_length, self.reaction_delay

        df = pd.read_parquet(fpath)
        df = df[df["frame"] >= 0].reset_index(drop=True)

        # ---- Encode c-stick direction
        for p in ("self", "opp", "self_nana", "opp_nana"):
            encode_cstick_dir(df, p, dead_zone=0.15)

        # ---- Drop unused & basic fillna
        df = df.drop(columns=["startAt"], errors="ignore")

        # ---- Recompute distance explicitly (using true on‐stage positions)
        df["distance"] = np.hypot(
            df["self_pos_x"] - df["opp_pos_x"],
            df["self_pos_y"] - df["opp_pos_y"],
        ).astype("float32")

        # ---- Fill defaults
        num_cols  = [c for c, dt in df.dtypes.items() if dt.kind in ("i", "f")]
        bool_cols = [c for c, dt in df.dtypes.items() if dt == "bool"]
        df[num_cols]  = df[num_cols].fillna(0.0)
        df[bool_cols] = df[bool_cols].fillna(False)

        # ---- Clamp only the exact numeric-features subset, with dtype coercion
        numeric_cols = []
        for _, meta in self._walk_groups(return_meta=True):
            if meta["ftype"] == "numeric":
                numeric_cols.extend(meta["cols"])
        numeric_cols = [c for c in set(numeric_cols) if c in df.columns]

        # **Fix**: cast to float32 before isfinite
        arr = df[numeric_cols].astype(np.float32).to_numpy()
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = 0.0
            df.loc[:, numeric_cols] = arr

        # ---- Map categoricals → ids (with guards) ----
        for c in self._categorical_cols:
            raw = df[c].fillna(0)
            df[c] = raw.map(lambda x: self._get_enum_map(c).get(x, 0)).astype("int64")
            if df[c].isna().any():
                bad = df.loc[df[c].isna(), c][:10].tolist()
                raise ValueError(f"{c}: unmapped values → {bad}")
            vocab_size = max(self._get_enum_map(c).values()) + 1
            if (df[c] >= vocab_size).any():
                max_bad = int(df[c].max())
                raise ValueError(f"{c}: index {max_bad} ≥ vocab {vocab_size}")

        # ---- Synthetic Nana presence flags ----
        df = df.assign(
            self_nana_present=(df.get("self_nana_character", 0) > 0).astype("float32"),
            opp_nana_present =(df.get("opp_nana_character", 0) > 0).astype("float32"),
        )

        # ---- Final guard (reuses arr, so no dtype issue) ----
        if not np.isfinite(arr).all():
            bad = [numeric_cols[i] for i in np.where(~np.isfinite(arr).any(axis=0))[0]]
            raise ValueError(f"Non-finite still in numeric: {bad[:5]}")

        # -----------------------------------------------------------------
        # Slice window + target
        # -----------------------------------------------------------------
        end_idx    = start_idx + W
        target_idx = end_idx + R - 1
        slice_df   = df.iloc[start_idx:end_idx].reset_index(drop=True)
        target_row = df.iloc[target_idx]

        # ---------- Build state_seq ----------
        state_seq: Dict[str, torch.Tensor] = {}
        for _, meta in self._walk_groups(return_meta=True):
            cols, ftype, entity = meta["cols"], meta["ftype"], meta["entity"]
            key = f"{entity}_{ftype}" if entity != "global" else ftype

            if ftype == "categorical":
                for col in cols:
                    state_seq[col] = torch.from_numpy(slice_df[col].values).long()
            else:
                arrs = [
                    torch.from_numpy(slice_df[col].astype("float32").values)
                    for col in cols
                ]
                state_seq[key] = (
                    torch.stack(arrs, dim=-1) if len(arrs) > 1 else arrs[0]
                )

        # ---------- Build target ----------
        target: Dict[str, torch.Tensor] = {
            "main_x":   torch.tensor(target_row["self_main_x"], dtype=torch.float32),
            "main_y":   torch.tensor(target_row["self_main_y"], dtype=torch.float32),
            "l_shldr":  torch.tensor(target_row["self_l_shldr"], dtype=torch.float32),
            "r_shldr":  torch.tensor(target_row["self_r_shldr"], dtype=torch.float32),
        }

        dir_idx = int(target_row["self_c_dir"])
        target["c_dir"] = torch.tensor(
            [1.0 if i == dir_idx else 0.0 for i in range(5)], dtype=torch.float32
        )

        target["btns"] = torch.stack(
            [
                torch.tensor(target_row[c], dtype=torch.float32)
                for c in btn_cols("self")
            ],
            dim=0,
        )

        return state_seq, target
