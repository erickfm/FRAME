#!/usr/bin/env python3
# dataset.py  –  Slippi parquet → fixed-length windows for FRAME v2
# -----------------------------------------------------------------------------
# • One authoritative TOKEN_SPEC that mirrors the 8 concept-tokens.
# • No hidden numeric bundles: each float/bool lives in a clearly named tensor.
# • Vectorised NumPy slice → torch.from_numpy → fast collation.
# • Still supports reaction-delay targets.
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------------
# Enum maps (fill these from your own cat_maps module or hard-code)
# -------------------------------------------------------------------------
from cat_maps import STAGE_MAP, CHARACTER_MAP, ACTION_MAP, PROJECTILE_TYPE_MAP

# -------------------------------------------------------------------------
# 0.  Helpers for column name patterns
# -------------------------------------------------------------------------
BTN = [
    "BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y",
    "BUTTON_Z", "BUTTON_L", "BUTTON_R", "BUTTON_START",
    "BUTTON_D_UP", "BUTTON_D_DOWN", "BUTTON_D_LEFT", "BUTTON_D_RIGHT",
]

def btn_cols(p: str)     -> List[str]: return [f"{p}_btn_{b}" for b in BTN]
def analog_cols(p: str)  -> List[str]: return [f"{p}_{k}" for k in
                                               ("main_x","main_y","l_shldr","r_shldr")]

# -------------------------------------------------------------------------
# 1.  Authoritative tensor spec  (mirrors FrameEncoder tokens)
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorSpec:
    cols:   List[str]      # raw column names, in order
    dtype:  str            # "float32", "int64", "bool"
    shape:  Tuple[int,...] # per-frame shape, e.g. (4,) or () for scalar
    token:  str            # GAME_STATE, SELF_INPUT, ...

# --- per-frame geometry -------------------------------------------------
_STAGE_GEOM = [
    "blastzone_left", "blastzone_right", "blastzone_top", "blastzone_bottom",
    "stage_edge_left", "stage_edge_right",
    "left_platform_height",  "left_platform_left",  "left_platform_right",
    "right_platform_height", "right_platform_left", "right_platform_right",
    "top_platform_height",   "top_platform_left",   "top_platform_right",
]

# full spec --------------------------------------------------------------
TOKEN_SPEC: Dict[str, TensorSpec] = {
    # ─── GAME_STATE ──────────────────────────────────────────────────
    "stage":         TensorSpec(["stage"],            "int64",   (),    "GAME_STATE"),
    "randall_state": TensorSpec(["randall_state"],    "int64",   (),    "GAME_STATE"),
    "stage_geom":    TensorSpec(_STA_GEOM := _STAGE_GEOM, "float32", (len(_STAGE_GEOM),), "GAME_STATE"),
    "distance":      TensorSpec(["distance"],         "float32", (),    "GAME_STATE"),

    # ─── SELF / OPP INPUTS ───────────────────────────────────────────
    "self_c_dir":    TensorSpec(["self_c_dir"],       "int64",   (),    "SELF_INPUT"),
    "self_sticks":   TensorSpec(analog_cols("self") + analog_cols("opp"),
                                "float32", (8,),      "SELF_INPUT"),
    "self_buttons":  TensorSpec(btn_cols("self"),     "bool",    (12,), "SELF_INPUT"),

    "opp_c_dir":     TensorSpec(["opp_c_dir"],        "int64",   (),    "OPP_INPUT"),
    "opp_sticks":    TensorSpec(analog_cols("opp") + analog_cols("self"),
                                "float32", (8,),      "OPP_INPUT"),
    "opp_buttons":   TensorSpec(btn_cols("opp"),      "bool",    (12,), "OPP_INPUT"),

    # ─── SELF / OPP STATE ────────────────────────────────────────────
    "self_port":        TensorSpec(["self_port"],     "int64",   (),    "SELF_STATE"),
    "self_character":   TensorSpec(["self_character"],"int64",   (),    "SELF_STATE"),
    "self_action":      TensorSpec(["self_action"],   "int64",   (),    "SELF_STATE"),
    "self_costume":     TensorSpec(["self_costume"],  "int64",   (),    "SELF_STATE"),
    "self_facing":      TensorSpec(["self_facing"],   "int64",   (),    "SELF_STATE"),
    "self_state_floats":TensorSpec([
        # 23 numeric features – adjust to your exact schema
        "self_pos_x","self_pos_y","self_percent","self_stock","self_jumps_left",
        "self_speed_air_x_self","self_speed_ground_x_self","self_speed_x_attack",
        "self_speed_y_attack","self_speed_y_self","self_hitlag_left","self_hitstun_left",
        "self_invuln_left","self_shield_strength",
        "self_ecb_bottom_x","self_ecb_bottom_y","self_ecb_left_x","self_ecb_left_y",
        "self_ecb_right_x","self_ecb_right_y","self_ecb_top_x","self_ecb_top_y",
        "self_action_frame",  # include elapsed frame
    ], "float32", (23,), "SELF_STATE"),
    "self_state_flags": TensorSpec([
        "self_on_ground","self_off_stage","self_invulnerable",
        "self_moonwalkwarning","self_stale_move_queue_not_empty",
        "self_is_in_hitlag","self_is_in_hitstun","self_is_tumbling","self_touching_shield",
    ], "bool", (9,), "SELF_STATE"),

    "opp_port":        TensorSpec(["opp_port"],     "int64", (), "OPP_STATE"),
    "opp_character":   TensorSpec(["opp_character"],"int64", (), "OPP_STATE"),
    "opp_action":      TensorSpec(["opp_action"],   "int64", (), "OPP_STATE"),
    "opp_costume":     TensorSpec(["opp_costume"],  "int64", (), "OPP_STATE"),
    "opp_facing":      TensorSpec(["opp_facing"],   "int64", (), "OPP_STATE"),
    "opp_state_floats":TensorSpec([
        "opp_pos_x","opp_pos_y","opp_percent","opp_stock","opp_jumps_left",
        "opp_speed_air_x_self","opp_speed_ground_x_self","opp_speed_x_attack",
        "opp_speed_y_attack","opp_speed_y_self","opp_hitlag_left","opp_hitstun_left",
        "opp_invuln_left","opp_shield_strength",
        "opp_ecb_bottom_x","opp_ecb_bottom_y","opp_ecb_left_x","opp_ecb_left_y",
        "opp_ecb_right_x","opp_ecb_right_y","opp_ecb_top_x","opp_ecb_top_y",
        "opp_action_frame",
    ], "float32", (23,), "OPP_STATE"),
    "opp_state_flags": TensorSpec([
        "opp_on_ground","opp_off_stage","opp_invulnerable",
        "opp_moonwalkwarning","opp_stale_move_queue_not_empty",
        "opp_is_in_hitlag","opp_is_in_hitstun","opp_is_tumbling","opp_touching_shield",
    ], "bool", (9,), "OPP_STATE"),

    # ─── NANA SELF / OPP ─────────────────────────────────────────────
    "self_nana_character": TensorSpec(["self_nana_character"], "int64", (), "NANA_SELF"),
    "self_nana_action":    TensorSpec(["self_nana_action"],    "int64", (), "NANA_SELF"),
    "self_nana_c_dir":     TensorSpec(["self_nana_c_dir"],     "int64", (), "NANA_SELF"),
    "self_nana_floats":    TensorSpec([
        # 24 floats
        "self_nana_pos_x","self_nana_pos_y","self_nana_percent","self_nana_stock",
        "self_nana_jumps_left","self_nana_speed_air_x_self","self_nana_speed_ground_x_self",
        "self_nana_speed_x_attack","self_nana_speed_y_attack","self_nana_speed_y_self",
        "self_nana_hitlag_left","self_nana_hitstun_left","self_nana_invuln_left",
        "self_nana_shield_strength",
        "self_nana_ecb_bottom_x","self_nana_ecb_bottom_y","self_nana_ecb_left_x",
        "self_nana_ecb_left_y","self_nana_ecb_right_x","self_nana_ecb_right_y",
        "self_nana_ecb_top_x","self_nana_ecb_top_y",
        "self_nana_action_frame",
        "self_nana_present",
    ], "float32", (24,), "NANA_SELF"),
    "self_nana_flags": TensorSpec([
        "self_nana_on_ground","self_nana_off_stage","self_nana_invulnerable",
        "self_nana_moonwalkwarning","self_nana_is_in_hitlag","self_nana_is_in_hitstun",
        "self_nana_is_tumbling","self_nana_touching_shield","self_nana_special_state",
        "self_nana_airborne","self_nana_fastfall","self_nana_hurtbox_state",
        "self_nana_power_shielding","self_nana_projectile","self_nana_reflecting",
        "self_nana_absorbing","self_nana_parrying",
    ], "bool", (17,), "NANA_SELF"),

    # (mirror for opp_nana …)
    "opp_nana_character": TensorSpec(["opp_nana_character"], "int64", (), "NANA_OPP"),
    "opp_nana_action":    TensorSpec(["opp_nana_action"],    "int64", (), "NANA_OPP"),
    "opp_nana_c_dir":     TensorSpec(["opp_nana_c_dir"],     "int64", (), "NANA_OPP"),
    "opp_nana_floats":    TensorSpec([
        "opp_nana_pos_x","opp_nana_pos_y","opp_nana_percent","opp_nana_stock",
        "opp_nana_jumps_left","opp_nana_speed_air_x_self","opp_nana_speed_ground_x_self",
        "opp_nana_speed_x_attack","opp_nana_speed_y_attack","opp_nana_speed_y_self",
        "opp_nana_hitlag_left","opp_nana_hitstun_left","opp_nana_invuln_left",
        "opp_nana_shield_strength",
        "opp_nana_ecb_bottom_x","opp_nana_ecb_bottom_y","opp_nana_ecb_left_x",
        "opp_nana_ecb_left_y","opp_nana_ecb_right_x","opp_nana_ecb_right_y",
        "opp_nana_ecb_top_x","opp_nana_ecb_top_y",
        "opp_nana_action_frame",
        "opp_nana_present",
    ], "float32", (24,), "NANA_OPP"),
    "opp_nana_flags": TensorSpec([
        "opp_nana_on_ground","opp_nana_off_stage","opp_nana_invulnerable",
        "opp_nana_moonwalkwarning","opp_nana_is_in_hitlag","opp_nana_is_in_hitstun",
        "opp_nana_is_tumbling","opp_nana_touching_shield","opp_nana_special_state",
        "opp_nana_airborne","opp_nana_fastfall","opp_nana_hurtbox_state",
        "opp_nana_power_shielding","opp_nana_projectile","opp_nana_reflecting",
        "opp_nana_absorbing","opp_nana_parrying",
    ], "bool", (17,), "NANA_OPP"),
}

# PROJECTILE tensors (categorical + 5 floats) ---------------------------
for slot in range(8):
    TOKEN_SPEC.update({
        f"proj{slot}_type":  TensorSpec([f"proj{slot}_type"],  "int64", (),       "PROJECTILES"),
        f"proj{slot}_subtype": TensorSpec([f"proj{slot}_subtype"], "int64", (),   "PROJECTILES"),
        f"proj{slot}_owner": TensorSpec([f"proj{slot}_owner"], "int64", (),       "PROJECTILES"),
        f"proj{slot}_floats":TensorSpec([
            f"proj{slot}_pos_x", f"proj{slot}_pos_y",
            f"proj{slot}_speed_x", f"proj{slot}_speed_y",
            f"proj{slot}_frame",
        ], "float32", (5,), "PROJECTILES"),
    })

# -------------------------------------------------------------------------
# 2.  C-stick direction encoding helper
# -------------------------------------------------------------------------
def encode_cstick_dir(df: pd.DataFrame, prefix: str, dead_zone: float = 0.15) -> str:
    dx = df[f"{prefix}_c_x"].astype("float32") - 0.5
    dy = df[f"{prefix}_c_y"].astype("float32") - 0.5
    mag = np.hypot(dx, dy)
    cat = np.zeros_like(mag, dtype="int64")
    alive = mag > dead_zone
    horiz, vert = alive & (np.abs(dx) >= np.abs(dy)), alive & (np.abs(dy) > np.abs(dx))
    cat[horiz & (dx > 0)] = 4; cat[horiz & (dx < 0)] = 3
    cat[vert  & (dy > 0)] = 1; cat[vert  & (dy < 0)] = 2
    new_col = f"{prefix}_c_dir"
    df[new_col] = cat
    return new_col

# -------------------------------------------------------------------------
# 3.  Main dataset
# -------------------------------------------------------------------------
class MeleeFrameDataset(Dataset):
    """Fixed-length windows over Slippi replays with reaction-delay targets."""

    def __init__(
        self,
        parquet_dir: str,
        sequence_length: int = 60,
        reaction_delay: int = 1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.reaction_delay  = reaction_delay
        self.parquet_dir     = Path(parquet_dir)
        self.files           = sorted(self.parquet_dir.glob("*.parquet"))
        if not self.files:
            raise RuntimeError(f"No .parquet files in {parquet_dir}")

        # map each possible starting frame across all files
        self.index_map: List[Tuple[Path, int]] = []
        for f in self.files:
            df = pd.read_parquet(f)
            df = df[df["frame"] >= 0]
            max_start = len(df) - (sequence_length + reaction_delay)
            if max_start > 0:
                self.index_map.extend([(f, s) for s in range(max_start)])

        # fixed enum maps
        self.enum_maps = {
            "stage": STAGE_MAP,
            "_character": CHARACTER_MAP,
            "_action":    ACTION_MAP,
            "_type":      PROJECTILE_TYPE_MAP,
            "_c_dir":     {i: i for i in range(5)},
        }

    # ---------------------------------------------------------------------
    def __len__(self) -> int:            return len(self.index_map)

    # ---------------------------------------------------------------------
    def _enum_map(self, col: str) -> Dict[int, int]:
        if col == "stage":
            return self.enum_maps["stage"]
        for suffix, m in self.enum_maps.items():
            if suffix != "stage" and col.endswith(suffix):
                return m
        raise KeyError(col)

    # ---------------------------------------------------------------------
    def __getitem__(self, idx: int):
        fpath, start = self.index_map[idx]
        W, R = self.sequence_length, self.reaction_delay

        df = pd.read_parquet(fpath)
        df = df[df["frame"] >= 0].reset_index(drop=True)

        # ---- encode C-stick directions
        for p in ("self", "opp", "self_nana", "opp_nana"):
            encode_cstick_dir(df, p)

        # ---- compute distance
        df["distance"] = np.hypot(
            df["self_pos_x"] - df["opp_pos_x"],
            df["self_pos_y"] - df["opp_pos_y"],
        ).astype("float32")

        # ---- randall_state (0 absent)
        df["randall_state"] = 0

        # ---- fill NAs
        df = df.fillna(0.0)

        # ---- map categoricals
        for name, spec in TOKEN_SPEC.items():
            if spec.dtype == "int64":
                raw = df[spec.cols[0]].astype("int64")
                mapper = self._enum_map(spec.cols[0])
                df[spec.cols[0]] = raw.map(lambda v: mapper.get(v, 0)).astype("int64")

        # ---- slice window + target frame
        end     = start + W
        tgt_idx = end + R - 1
        win_df  = df.iloc[start:end].reset_index(drop=True)
        tgt_row = df.iloc[tgt_idx]

        # -----------------------------------------------------------------
        # Build STATE dict
        # -----------------------------------------------------------------
        state: Dict[str, torch.Tensor] = {}
        for name, spec in TOKEN_SPEC.items():
            arr = win_df[spec.cols].to_numpy(spec.dtype)
            # boolean → float32  (easier for linear layers later)
            if spec.dtype == "bool":
                arr = arr.astype("float32")
            if arr.ndim == 1:
                arr = arr[:, None] if spec.shape else arr
            tensor = torch.from_numpy(arr)  # (T, C) or (T,)
            state[name] = tensor

        # -----------------------------------------------------------------
        # Build TARGET dict  (unchanged from your training loop)
        # -----------------------------------------------------------------
        target = {
            "main_x":  torch.tensor(tgt_row["self_main_x"],  dtype=torch.float32),
            "main_y":  torch.tensor(tgt_row["self_main_y"],  dtype=torch.float32),
            "l_shldr": torch.tensor(tgt_row["self_l_shldr"], dtype=torch.float32),
            "r_shldr": torch.tensor(tgt_row["self_r_shldr"], dtype=torch.float32),
            "c_dir":   torch.nn.functional.one_hot(
                torch.tensor(int(tgt_row["self_c_dir"]), dtype=torch.long),
                num_classes=5,
            ).float(),
            "btns": torch.tensor(
                tgt_row[btn_cols("self")].to_numpy("float32"), dtype=torch.float32
            ),
        }

        return state, target

# -------------------------------------------------------------------------
# 4.  Collate fn (simple because every tensor already (B,T,…))
# -------------------------------------------------------------------------
def collate_fn(batch):
    state, tgt = {}, {}
    for k in TOKEN_SPEC.keys():
        state[k] = torch.stack([item[0][k] for item in batch], 0)  # (B,T,…)
    for k in batch[0][1]:
        tgt[k] = torch.stack([item[1][k] for item in batch], 0)
    return state, tgt
