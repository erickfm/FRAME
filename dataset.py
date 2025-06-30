#!/usr/bin/env python3
# dataset.py – Slippi parquet → fixed-length windows for FRAME (2025-06 spec)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ------------------------------------------------------------------
# 0. Enum maps  (swap with real dicts if needed)
# ------------------------------------------------------------------
from cat_maps import STAGE_MAP, CHARACTER_MAP, ACTION_MAP, PROJECTILE_TYPE_MAP

# ------------------------------------------------------------------
# 1. Column helpers
# ------------------------------------------------------------------
BTN = [
    "BUTTON_A","BUTTON_B","BUTTON_X","BUTTON_Y","BUTTON_Z","BUTTON_L",
    "BUTTON_R","BUTTON_START","BUTTON_D_UP","BUTTON_D_DOWN",
    "BUTTON_D_LEFT","BUTTON_D_RIGHT",
]
def btn_cols(p:str)->List[str]:    return [f"{p}_btn_{b}" for b in BTN]
def analog_cols(p:str)->List[str]: return [f"{p}_{k}" for k in
                                           ("main_x","main_y","l_shldr","r_shldr")]

# ------------------------------------------------------------------
# 2. TensorSpec
# ------------------------------------------------------------------
@dataclass(frozen=True)
class TensorSpec:
    cols:  List[str]
    dtype: str              # "float32" | "int64" | "bool"
    shape: Tuple[int, ...]
    token: str

STAGE_GEOM = [
    "blastzone_left","blastzone_right","blastzone_top","blastzone_bottom",
    "stage_edge_left","stage_edge_right",
    "left_platform_height","left_platform_left","left_platform_right",
    "right_platform_height","right_platform_left","right_platform_right",
    "top_platform_height","top_platform_left","top_platform_right",
]                                       # 15 floats

PLAYER_FLAGS = ["on_ground","off_stage","invulnerable","moonwalkwarning","facing"]
def flag_cols(p:str): return [f"{p}_{f}" for f in PLAYER_FLAGS]

def player_state_cols(p:str)->List[str]:
    return [
        f"{p}_pos_x", f"{p}_pos_y",
        f"{p}_percent", f"{p}_stock", f"{p}_jumps_left",
        f"{p}_speed_air_x_self", f"{p}_speed_ground_x_self",
        f"{p}_speed_x_attack", f"{p}_speed_y_attack", f"{p}_speed_y_self",
        f"{p}_hitlag_left", f"{p}_hitstun_left", f"{p}_invuln_left",
        f"{p}_shield_strength",
        f"{p}_ecb_bottom_x", f"{p}_ecb_bottom_y",
        f"{p}_ecb_left_x",  f"{p}_ecb_left_y",
        f"{p}_ecb_right_x", f"{p}_ecb_right_y",
        f"{p}_ecb_top_x",   f"{p}_ecb_top_y",
        f"{p}_action_frame",
    ]                                    # 23 floats

def nana_state_cols(p:str)->List[str]:
    return player_state_cols(p)[:-1] + [f"{p}_action_frame", f"{p}_present"]  # 24

# ------------- build TOKEN_SPEC dict --------------------------------
TOKEN_SPEC: Dict[str, TensorSpec] = {
    # GAME_STATE
    "stage":       TensorSpec(["stage"], "int64", (), "GAME_STATE"),
    "stage_geom":  TensorSpec(STAGE_GEOM, "float32",(15,), "GAME_STATE"),
    "randall":     TensorSpec(["randall_height","randall_left","randall_right"],
                              "float32",(3,), "GAME_STATE"),
    "distance":    TensorSpec(["distance"], "float32", (), "GAME_STATE"),
    "frame":       TensorSpec(["frame"], "float32", (), "GAME_STATE"),

    # SELF / OPP INPUT
    "self_c_dir":  TensorSpec(["self_c_dir"], "int64", (), "SELF_INPUT"),
    "self_sticks": TensorSpec(analog_cols("self")+analog_cols("opp"),
                              "float32",(8,), "SELF_INPUT"),
    "self_buttons":TensorSpec(btn_cols("self"), "bool",(12,), "SELF_INPUT"),

    "opp_c_dir":   TensorSpec(["opp_c_dir"], "int64", (), "OPP_INPUT"),
    "opp_sticks":  TensorSpec(analog_cols("opp")+analog_cols("self"),
                              "float32",(8,), "OPP_INPUT"),
    "opp_buttons": TensorSpec(btn_cols("opp"), "bool",(12,), "OPP_INPUT"),
}

for side in ("self","opp"):
    TOKEN_SPEC.update({
        f"{side}_port":        TensorSpec([f"{side}_port"], "int64", (), f"{side.upper()}_STATE"),
        f"{side}_character":   TensorSpec([f"{side}_character"],"int64", (), f"{side.upper()}_STATE"),
        f"{side}_action":      TensorSpec([f"{side}_action"],   "int64", (), f"{side.upper()}_STATE"),
        f"{side}_costume":     TensorSpec([f"{side}_costume"],  "int64", (), f"{side.upper()}_STATE"),
        f"{side}_state_floats":TensorSpec(player_state_cols(side),"float32",(23,), f"{side.upper()}_STATE"),
        f"{side}_state_flags": TensorSpec(flag_cols(side), "bool",(5,), f"{side.upper()}_STATE"),
    })

for nana, token in (("self_nana","NANA_SELF"),("opp_nana","NANA_OPP")):
    TOKEN_SPEC.update({
        f"{nana}_character": TensorSpec([f"{nana}_character"],"int64", (), token),
        f"{nana}_action":    TensorSpec([f"{nana}_action"],   "int64", (), token),
        f"{nana}_c_dir":     TensorSpec([f"{nana}_c_dir"],    "int64", (), token),
        f"{nana}_floats":    TensorSpec(nana_state_cols(nana),"float32",(24,), token),
        f"{nana}_flags":     TensorSpec(flag_cols(nana), "bool",(5,),   token),
    })

for j in range(8):
    TOKEN_SPEC.update({
        f"proj{j}_type":    TensorSpec([f"proj{j}_type"],   "int64", (), "PROJECTILES"),
        f"proj{j}_subtype": TensorSpec([f"proj{j}_subtype"],"int64", (), "PROJECTILES"),
        f"proj{j}_owner":   TensorSpec([f"proj{j}_owner"],  "int64", (), "PROJECTILES"),
        f"proj{j}_floats":  TensorSpec([
            f"proj{j}_pos_x", f"proj{j}_pos_y",
            f"proj{j}_speed_x", f"proj{j}_speed_y",
            f"proj{j}_frame",
        ], "float32",(5,), "PROJECTILES"),
    })

# ------------------------------------------------------------------
# 3. C-stick encoder
# ------------------------------------------------------------------
def encode_cstick_dir(df: pd.DataFrame, prefix: str, dead: float = 0.15):
    dx = df[f"{prefix}_c_x"].astype("float32") - 0.5
    dy = df[f"{prefix}_c_y"].astype("float32") - 0.5
    mag = np.hypot(dx, dy)
    cat = np.zeros_like(mag, dtype="int64")
    alive = mag > dead
    horiz = alive & (np.abs(dx) >= np.abs(dy))
    vert  = alive & (np.abs(dy) >  np.abs(dx))
    cat[horiz & (dx > 0)] = 4; cat[horiz & (dx < 0)] = 3
    cat[vert  & (dy > 0)] = 1; cat[vert  & (dy < 0)] = 2
    df[f"{prefix}_c_dir"] = cat

# ------------------------------------------------------------------
# 4. Dataset
# ------------------------------------------------------------------
class MeleeFrameDataset(Dataset):
    def __init__(self, parquet_dir: str, seq_len: int = 60, delay: int = 1, debug: bool = False):
        super().__init__()
        self.W, self.D, self.debug = seq_len, delay, debug

        self.files = sorted(Path(parquet_dir).glob("*.parquet"))
        self.index: List[Tuple[Path, int]] = []
        for f in self.files:
            df = pd.read_parquet(f)
            df = df[df["frame"] >= 0]
            max_start = len(df) - (seq_len + delay)
            if max_start > 0:
                self.index.extend([(f, s) for s in range(max_start)])

        self.enums: Dict[str, Dict[int, int]] = {
            "stage":      STAGE_MAP,
            "_character": CHARACTER_MAP,
            "_action":    ACTION_MAP,
            "_type":      PROJECTILE_TYPE_MAP,
            "_subtype":   {i: i for i in range(256)},
            "_c_dir":     {i: i for i in range(5)},
            "_port":      {i: i for i in range(4)},
            "_owner":     {i: i for i in range(4)},
            "_costume":   {i: i for i in range(8)},
        }

    # —— helpers ————————————————————————————————————————————————
    def __len__(self): return len(self.index)

    def _enum(self, col: str) -> Dict[int, int]:
        if col == "stage": return self.enums["stage"]
        for suff, m in self.enums.items():
            if suff != "stage" and col.endswith(suff):
                return m
        return {v: v for v in range(int(self._current_df[col].max()) + 1)}  # identity fallback

    def _tensor(self, df: pd.DataFrame, spec: TensorSpec) -> torch.Tensor:
        T = len(df)
        if spec.shape == ():  # scalar
            col = spec.cols[0]
            arr = df[col].to_numpy(spec.dtype) if col in df else np.zeros(T, spec.dtype)
        else:
            C = spec.shape[0]
            arr = np.zeros((T, C), spec.dtype)
            for i, col in enumerate(spec.cols):
                if col in df:
                    arr[:, i] = df[col].astype(spec.dtype).to_numpy()
        if spec.dtype == "bool":
            arr = arr.astype("float32")

        # final finite check
        bad = ~np.isfinite(arr) if arr.dtype.kind == "f" else None
        if bad is not None and bad.any():
            if self.debug:
                print(f"⚠️ patched non-finite in {spec.token}/{spec.cols[0]}")
            arr[bad] = 0.0
        return torch.from_numpy(arr)

    # —— main loader ——————————————————————————————————————————————
    def __getitem__(self, idx: int):
        path, start = self.index[idx]
        df = pd.read_parquet(path).drop(columns=["startAt"], errors="ignore")
        df = df[df["frame"] >= 0].reset_index(drop=True)

        # 1. derived
        for p in ("self", "opp", "self_nana", "opp_nana"):
            encode_cstick_dir(df, p)
        df["distance"] = np.hypot(df["self_pos_x"] - df["opp_pos_x"],
                                  df["self_pos_y"] - df["opp_pos_y"]).astype("float32")
        df["self_nana_present"] = (df["self_nana_character"] > 0).astype("float32")
        df["opp_nana_present"]  = (df["opp_nana_character"]  > 0).astype("float32")
        for c in ("self_facing", "opp_facing", "self_nana_facing", "opp_nana_facing"):
            df[c] = (df[c] > 0).astype("float32")

        # 2. << SIMPLE numeric coercion block >>
        float_cols = [
            c
            for spec in TOKEN_SPEC.values() if spec.dtype == "float32"
            for c in spec.cols if c in df.columns
        ]
        if float_cols:
            df[float_cols] = (
                df[float_cols]
                .apply(pd.to_numeric, errors="coerce")   # None/text → NaN
                .fillna(0.0)                             # NaN       → 0.0
                .astype("float32")
            )

        # 3. enum mapping
        self._current_df = df  # for fallback
        for spec in TOKEN_SPEC.values():
            if spec.dtype == "int64":
                col = spec.cols[0]
                if col in df:
                    mp = self._enum(col)
                    df[col] = df[col].map(lambda v: mp.get(int(v), 0)).astype("int64")

        # 4. slice window & target
        win = df.iloc[start:start + self.W].reset_index(drop=True)
        tgt = df.iloc[start + self.W + self.D - 1]

        state = {k: self._tensor(win, spec) for k, spec in TOKEN_SPEC.items()}

        target = {
            "main_x":  torch.tensor(tgt["self_main_x"],  dtype=torch.float32),
            "main_y":  torch.tensor(tgt["self_main_y"],  dtype=torch.float32),
            "l_shldr": torch.tensor(tgt["self_l_shldr"], dtype=torch.float32),
            "r_shldr": torch.tensor(tgt["self_r_shldr"], dtype=torch.float32),
            "c_dir": torch.nn.functional.one_hot(
                torch.tensor(int(tgt["self_c_dir"]), dtype=torch.long), 5
            ).float(),
            "btns": torch.tensor(
                tgt[btn_cols("self")].to_numpy("float32"), dtype=torch.float32),
        }
        return state, target

# ------------------------------------------------------------------
# 5. Collate
# ------------------------------------------------------------------
def collate_fn(batch):
    state, target = {}, {}
    for k in TOKEN_SPEC:
        state[k] = torch.stack([b[0][k] for b in batch], 0)
    for k in batch[0][1]:
        target[k] = torch.stack([b[1][k] for b in batch], 0)
    return state, target
