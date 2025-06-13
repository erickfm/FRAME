#!/usr/bin/env python3
# inference.py
#
# Real-time FRAME bot:
#   • Converts live Slippi frames to dataset tensors
#   • Encodes c-stick floats → 5-way categorical direction
#   • Runs FramePredictor on a rolling window
#   • Converts model output back to Dolphin controller actions
# ---------------------------------------------------------------------------

import math
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import melee

# ── maps & model ------------------------------------------------------------
from cat_maps import STAGE_MAP, CHARACTER_MAP, ACTION_MAP, PROJECTILE_TYPE_MAP
from model    import FramePredictor, ModelConfig
from dataset  import MeleeFrameDatasetWithDelay  # only for feature spec

# ════════════════════════════════════════════════════════════════════════════
# 0)  Device + checkpoint
# ════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

ckpts = sorted(Path("./checkpoints").glob("epoch_*.pt"))
if not ckpts:
    print("No checkpoints found."); sys.exit(1)
ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)
ckpt      = torch.load(ckpt_path, map_location=DEVICE)
cfg       = ModelConfig(**ckpt["config"])

model = FramePredictor(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Loaded", ckpt_path)

ROLL_WIN = cfg.max_seq_len
MAX_PROJ = 8

# ════════════════════════════════════════════════════════════════════════════
# 1)  Feature spec from Dataset
# ════════════════════════════════════════════════════════════════════════════
_spec = MeleeFrameDatasetWithDelay.__new__(MeleeFrameDatasetWithDelay)
_spec.feature_groups = _spec._build_feature_groups()
_spec._categorical_cols = [
    col for _, meta in _spec._walk_groups(return_meta=True)
    if meta["ftype"] == "categorical" for col in meta["cols"]
]
_spec._enum_maps = {
    "stage":      STAGE_MAP,
    "_character": CHARACTER_MAP,
    "_action":    ACTION_MAP,
    "_type":      PROJECTILE_TYPE_MAP,
    "c_dir":      {i: i for i in range(5)},  # identity map
}

# ════════════════════════════════════════════════════════════════════════════
# 2)  Helpers
# ════════════════════════════════════════════════════════════════════════════
def encode_cstick_dir_df(df: pd.DataFrame, prefix: str, dead: float = 0.15):
    dx = df[f"{prefix}_c_x"].astype(np.float32) - 0.5
    dy = 0.5 - df[f"{prefix}_c_y"].astype(np.float32)      # up = +Y
    mag = np.hypot(dx, dy)

    cat = np.zeros_like(mag, dtype=np.int64)
    active = mag > dead
    horiz  = active & (np.abs(dx) >= np.abs(dy))
    vert   = active & (np.abs(dy) >  np.abs(dx))

    cat[horiz & (dx > 0)] = 4
    cat[horiz & (dx < 0)] = 3
    cat[vert  & (dy > 0)] = 1
    cat[vert  & (dy < 0)] = 2
    df[f"{prefix}_c_dir"] = cat


def _map_cat(col: str, x: Any) -> int:
    if col == "stage":
        return _spec._enum_maps["stage"].get(x, 0)
    if col.endswith("_c_dir"):
        return int(x) if 0 <= int(x) <= 4 else 0
    for suf, mp in _spec._enum_maps.items():
        if suf != "stage" and col.endswith(suf):
            return mp.get(x, 0)
    try:
        return max(int(x), 0)
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════════════════════
# 3)  rows → state_seq
# ════════════════════════════════════════════════════════════════════════════
def rows_to_state_seq(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    df = pd.DataFrame(rows).drop(columns=["startAt"], errors="ignore")

    # encode c_dirs before any mapping
    for p in ("self", "opp", "self_nana", "opp_nana"):
        if f"{p}_c_x" in df.columns:
            encode_cstick_dir_df(df, p)

    # fill NaNs for numeric / bool
    num_cols  = [c for c, t in df.dtypes.items() if t.kind in ("i", "f")]
    bool_cols = [c for c, t in df.dtypes.items() if t == "bool"]
    df[num_cols]  = df[num_cols].fillna(0.0)
    df[bool_cols] = df[bool_cols].fillna(False)

    # ------------------------------------------------------------------
    # Guarantee ALL categorical columns exist – batch insert to avoid
    # fragmentation warnings
    # ------------------------------------------------------------------
    missing_cats = [c for c in _spec._categorical_cols if c not in df.columns]
    if missing_cats:
        df = pd.concat([df, pd.DataFrame({c: 0 for c in missing_cats},
                                         index=df.index)], axis=1)

    # map categoricals
    for col in _spec._categorical_cols:
        df[col] = df[col].map(lambda x, c=col: _map_cat(c, x)).astype("int64")

    # synthetic Nana flags
    df["self_nana_present"] = (df.get("self_nana_character", 0) > 0).astype("float32")
    df["opp_nana_present"]  = (df.get("opp_nana_character", 0) > 0).astype("float32")

    # ensure numeric cols exist (batch insert)
    numeric_missing = {}
    for _, meta in _spec._walk_groups(return_meta=True):
        if meta["ftype"] != "categorical":
            for col in meta["cols"]:
                if col not in df.columns:
                    numeric_missing[col] = 0.0
    if numeric_missing:
        df = pd.concat([df, pd.DataFrame(numeric_missing, index=df.index)], axis=1)

    # build tensor dict
    state_seq: Dict[str, torch.Tensor] = {}
    for _, meta in _spec._walk_groups(return_meta=True):
        cols, ftype, entity = meta["cols"], meta["ftype"], meta["entity"]
        key = f"{entity}_{ftype}" if entity != "global" else ftype
        if ftype == "categorical":
            for col in cols:
                state_seq[col] = torch.from_numpy(df[col].values).long().unsqueeze(0)
        else:
            mats = [torch.from_numpy(df[c].astype(np.float32).values) for c in cols]
            state_seq[key] = (torch.stack(mats, -1) if len(mats) > 1 else mats[0]).unsqueeze(0)
    return state_seq


@torch.no_grad()
def run_inference(win_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = rows_to_state_seq(win_rows)
    for k, v in batch.items():
        batch[k] = v.to(DEVICE)
    preds = model(batch)
    return {k: v.cpu().squeeze(0) for k, v in preds.items()}

# ════════════════════════════════════════════════════════════════════════════
# 4)  Controller output (with NaN-safe clamp)
# ════════════════════════════════════════════════════════════════════════════
C_DIR_TO_FLOAT = {
    0: (0.5, 0.5),
    1: (0.5, 1.0),
    2: (0.5, 0.0),
    3: (0.0, 0.5),
    4: (1.0, 0.5),
}

def _safe(val: float, default: float = 0.5) -> float:
    """Replace NaN/inf with default and clamp to [0,1]."""
    if not math.isfinite(val):
        return default
    return min(max(val, 0.0), 1.0)

def press_output(ctrl: melee.Controller,
                 pred: Dict[str, torch.Tensor],
                 thresh: float = 0.7):

    mx, my = map(float, pred["main_xy"].tolist())
    mx, my = _safe(mx), _safe(my)

    dir_idx = int(torch.argmax(pred["c_dir_logits"]))
    cx, cy  = C_DIR_TO_FLOAT.get(dir_idx, (0.5, 0.5))

    l_val = _safe(pred["L_val"].item(), 0.0)
    r_val = _safe(pred["R_val"].item(), 0.0)

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx, my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C,    cx, cy)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L, l_val)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_R, r_val)

    idx_to_button = [
        melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
        melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
        melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
        melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_START,
        melee.enums.Button.BUTTON_D_UP, melee.enums.Button.BUTTON_D_DOWN,
        melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,
    ]
    for prob, btn in zip(pred["btn_probs"], idx_to_button):
        (ctrl.press_button if prob.item() > thresh else ctrl.release_button)(btn)

# ════════════════════════════════════════════════════════════════════════════
# 5)  Dolphin loop
# ════════════════════════════════════════════════════════════════════════════
def signal_handler(sig, _):
    for c in controllers.values():
        c.disconnect()
    console.stop()
    print("Shutting down…")
    sys.exit(0)

if __name__ == "__main__":
    DOLPHIN_APP = "/Users/erick/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app"
    ISO_PATH    = "/Users/erick/Downloads/melee.iso"

    console = melee.Console(path=DOLPHIN_APP, slippi_address="127.0.0.1", fullscreen=False)
    ports = [1, 2]
    controllers = {p: melee.Controller(console, p) for p in ports}

    signal.signal(signal.SIGINT, signal_handler)
    console.run(iso_path=ISO_PATH)

    if not console.connect():
        print("Console connect failed"); sys.exit(1)
    for c in controllers.values():
        if not c.connect():
            print("Controller connect failed"); sys.exit(1)
    print("Console + controllers connected.")

    rows: deque[Dict[str, Any]] = deque(maxlen=ROLL_WIN)

    while True:
        gs = console.step()
        if gs is None:
            continue

        # menu helper (unchanged)
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            c0, c1 = controllers[ports[0]], controllers[ports[1]]
            melee.MenuHelper().menu_helper_simple(
                gs, c0, melee.Character.FALCO, melee.Stage.POKEMON_STADIUM,
                cpu_level=0, autostart=0
            )
            melee.MenuHelper().menu_helper_simple(
                gs, c1, melee.Character.FALCO, melee.Stage.POKEMON_STADIUM,
                cpu_level=1, autostart=1
            )
            continue

        # -------------------------------------------------------------------
        # Build a single-row dict (mirrors dataset column names)
        # -------------------------------------------------------------------
        row: Dict[str, Any] = {}
        row["stage"]    = gs.stage.name if gs.stage else ""
        row["frame"]    = gs.frame
        row["distance"] = gs.distance
        row["startAt"]  = gs.startAt

        for idx, (port, ps) in enumerate(gs.players.items()):
            pref = "self_" if idx == 0 else "opp_"
            row[f"{pref}port"]          = port
            row[f"{pref}character"]     = ps.character.name
            row[f"{pref}action"]        = ps.action.name
            row[f"{pref}action_frame"]  = ps.action_frame
            row[f"{pref}costume"]       = ps.costume

            for btn, st in ps.controller_state.button.items():
                row[f"{pref}btn_{btn.name}"] = int(st)

            row[f"{pref}main_x"], row[f"{pref}main_y"] = ps.controller_state.main_stick
            row[f"{pref}c_x"],   row[f"{pref}c_y"]    = ps.controller_state.c_stick
            row[f"{pref}l_shldr"] = ps.controller_state.l_shoulder
            row[f"{pref}r_shldr"] = ps.controller_state.r_shoulder

            row[f"{pref}percent"] = float(ps.percent)
            row[f"{pref}pos_x"]   = float(ps.position.x)
            row[f"{pref}pos_y"]   = float(ps.position.y)

            nana = ps.nana
            npref = f"{pref}nana_"
            if nana:
                row[f"{npref}character"]    = nana.character.name
                row[f"{npref}action"]       = nana.action.name
                row[f"{npref}action_frame"] = nana.action_frame
            else:
                row[f"{npref}character"]    = ""
                row[f"{npref}action"]       = ""
                row[f"{npref}action_frame"] = -1

        # projectiles
        for j in range(MAX_PROJ):
            pp = f"proj{j}_"
            if j < len(gs.projectiles):
                p = gs.projectiles[j]
                row[f"{pp}owner"]   = p.owner
                row[f"{pp}type"]    = p.type.name
                row[f"{pp}subtype"] = p.subtype
                row[f"{pp}pos_x"]   = float(p.position.x)
                row[f"{pp}pos_y"]   = float(p.position.y)
                row[f"{pp}speed_x"] = float(p.speed.x)
                row[f"{pp}speed_y"] = float(p.speed.y)
                row[f"{pp}frame"]   = p.frame
            else:
                row[f"{pp}owner"]   = -1
                row[f"{pp}type"]    = ""
                row[f"{pp}subtype"] = -1
                row[f"{pp}pos_x"]   = np.nan
                row[f"{pp}pos_y"]   = np.nan
                row[f"{pp}speed_x"] = np.nan
                row[f"{pp}speed_y"] = np.nan
                row[f"{pp}frame"]   = -1

        rows.append(row)
        if len(rows) == ROLL_WIN:
            pred = run_inference(list(rows))
            ctrl = controllers[ports[0]]
            ctrl.release_all()
            press_output(ctrl, pred)
            ctrl.flush()

            if gs.frame % 60 == 0:
                print(f"[{gs.frame}] main={pred['main_xy'].tolist()} "
                      f"cdir={int(torch.argmax(pred['c_dir_logits']))} "
                      f"btn0-3={pred['btn_probs'][:4].tolist()}")
