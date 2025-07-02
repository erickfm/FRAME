#!/usr/bin/env python3
# inference.py  –  FRAME bot (debug-enhanced)
#
# • Converts live Slippi frames to dataset tensors
# • Encodes c-stick floats → 5-way categorical direction
# • Runs FramePredictor on a rolling window
# • Converts model output back to Dolphin controller actions
# • Adds detailed sanity checks for NaNs, infs, dtypes & shapes
# ---------------------------------------------------------------------------

import argparse
import logging
import math
import os
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import melee
import numpy as np
import pandas as pd
import torch

# ── CLI / logging ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Realtime FRAME bot w/ debug")
parser.add_argument("--debug", action="store_true", help="Verbose sanity checks")
args = parser.parse_args()
DEBUG = args.debug or bool(os.getenv("DEBUG", ""))

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s"
)
log = logging.getLogger(__name__)
torch.set_printoptions(sci_mode=False, precision=4)

# ── maps & model ─────────────────────────────────────────────────────────────
from cat_maps import STAGE_MAP, CHARACTER_MAP, ACTION_MAP, PROJECTILE_TYPE_MAP
from model     import FramePredictor, ModelConfig
from dataset   import (
    MeleeFrameDataset,
    TOKEN_SPEC,
    encode_cstick_dir,
)

# ════════════════════════════════════════════════════════════════════════════
# 0)  Device + checkpoint
# ════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

ckpts = sorted(Path("./checkpoints").glob("epoch_*.pt"))
if not ckpts:
    log.error("No checkpoints found."); sys.exit(1)
ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)
ckpt      = torch.load(ckpt_path, map_location=DEVICE)
cfg       = ModelConfig(**ckpt["config"])

model = FramePredictor(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
log.info("Loaded checkpoint %s", ckpt_path)

ROLL_WIN = cfg.max_seq_len
MAX_PROJ = 8

# ════════════════════════════════════════════════════════════════════════════
# 1)  Feature spec from Dataset
# ════════════════════════════════════════════════════════════════════════════
_spec = MeleeFrameDataset.__new__(MeleeFrameDataset)
_spec.enums = {
    "stage": STAGE_MAP,
    "_character": CHARACTER_MAP,
    "_action": ACTION_MAP,
    "_type": PROJECTILE_TYPE_MAP,
    "_subtype": {i: i for i in range(42)},
    "_c_dir": {i: i for i in range(5)},
    "_port": {i: i for i in range(4)},
    "_owner": {i: i for i in range(4)},
    "_costume": {i: i for i in range(8)},
}
_categorical_cols = [col for name, spec in TOKEN_SPEC.items() if spec.dtype == "int64" for col in spec.cols]

# ════════════════════════════════════════════════════════════════════════════
# 2)  Debug helpers
# ════════════════════════════════════════════════════════════════════════════
def check_tensor_dict(tdict: Dict[str, torch.Tensor], where: str) -> None:
    """Warn if any NaN/Inf or unexpected dtype/shape sneaks in."""
    if not DEBUG:
        return
    for k, v in tdict.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            log.warning("NaN/Inf detected in %s → %s", where, k)
            bad_idx = torch.isnan(v) | torch.isinf(v)
            log.debug("Offending values [%s]: %s", k, v[bad_idx][:10])
        if v.device != DEVICE:
            log.debug("Tensor %s not on %s (on %s instead)", k, DEVICE, v.device)
        if v.dtype not in (torch.float32, torch.int64, torch.bool):
            log.debug("Unexpected dtype in %s → %s (%s)", where, k, v.dtype)

# ════════════════════════════════════════════════════════════════════════════
# 3)  dataframe utils
# ════════════════════════════════════════════════════════════════════════════
def encode_cstick_dir_df(df: pd.DataFrame, prefix: str, dead: float = 0.15) -> None:
    """Wrapper using dataset.encode_cstick_dir for live frames."""
    if f"{prefix}_c_x" in df.columns and f"{prefix}_c_y" in df.columns:
        encode_cstick_dir(df, prefix, dead)


def _map_cat(col: str, x: Any) -> int:
    """Map raw enum value or name to dataset index."""
    m = _spec._enum(col)
    try:
        if isinstance(x, str) and x:
            if col == "stage":
                x = melee.enums.Stage[x].value
            elif col.endswith("_character"):
                x = melee.enums.Character[x].value
            elif col.endswith("_action"):
                x = melee.enums.Action[x].value
            elif col.endswith("_type"):
                x = melee.enums.ProjectileType[x].value
        x = int(x)
    except Exception:
        x = 0
    return m.get(x, 0)


def rows_to_state_seq(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Convert rolling list of row dicts → tensor batch (B=1)."""
    df = pd.DataFrame(rows).drop(columns=["startAt"], errors="ignore")

    # ---------- C-stick dirs ----------
    for p in ("self", "opp", "self_nana", "opp_nana"):
        encode_cstick_dir_df(df, p)

    # ---------- Fill NaNs ----------
    num_cols  = [c for c, t in df.dtypes.items() if t.kind in ("i", "f")]
    bool_cols = [c for c, t in df.dtypes.items() if t == "bool"]
    df[num_cols]  = df[num_cols].fillna(0.0)
    df[bool_cols] = df[bool_cols].fillna(False)

    # ---------- Ensure categorical columns ----------
    missing_cats = [c for c in _categorical_cols if c not in df.columns]
    if missing_cats:
        df = pd.concat(
            [df, pd.DataFrame({c: 0 for c in missing_cats}, index=df.index)],
            axis=1,
        )

    # ---------- Map categoricals ----------
    for col in _categorical_cols:
        df[col] = df[col].map(lambda x, c=col: _map_cat(c, x)).astype("int64")

    # ---------- Derived fields ----------
    if "self_pos_x" in df.columns and "opp_pos_x" in df.columns:
        df["distance"] = np.hypot(
            df["self_pos_x"] - df["opp_pos_x"],
            df["self_pos_y"] - df["opp_pos_y"],
        ).astype("float32")
    df["self_nana_present"] = (df.get("self_nana_character", 0) > 0).astype("float32")
    df["opp_nana_present"]  = (df.get("opp_nana_character", 0) > 0).astype("float32")

    for col in [
        "self_facing",
        "opp_facing",
        "self_nana_facing",
        "opp_nana_facing",
    ]:
        if col in df.columns:
            df[col] = (df[col] > 0).astype("float32")

    # ---------- Ensure numeric columns ----------
    for name, spec in TOKEN_SPEC.items():
        if spec.dtype != "int64":
            for col in spec.cols:
                if col not in df.columns:
                    df[col] = 0.0

    # Optional dataframe NaN audit
    if DEBUG and df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        log.warning("DataFrame still has NaNs in cols: %s", bad)

    # ---------- Build tensor dict ----------
    state_seq: Dict[str, torch.Tensor] = {
        name: _spec._tensor(df, spec).unsqueeze(0)
        for name, spec in TOKEN_SPEC.items()
    }

    if DEBUG:
        check_tensor_dict(state_seq, "state_seq")
    return state_seq

# ════════════════════════════════════════════════════════════════════════════
# 4)  Inference wrapper
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(win_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = rows_to_state_seq(win_rows)
    for k, v in batch.items():
        batch[k] = v.to(DEVICE, non_blocking=True)

    check_tensor_dict(batch, "batch_before_model")
    preds = model(batch)
    check_tensor_dict(preds, "model_output")

    return {k: v.cpu().squeeze(0) for k, v in preds.items()}

# ════════════════════════════════════════════════════════════════════════════
# 5)  Controller output (with NaN-safe clamp)
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
                 thresh: float = 0.5):

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

    pressed = []
    for prob, btn in zip(pred["btn_probs"], idx_to_button):
        if prob.item() > thresh:
            ctrl.press_button(btn)
            pressed.append(btn.name)
        else:
            ctrl.release_button(btn)

    if DEBUG:
        log.debug(
            "press_output → main=(%.2f,%.2f) c=(%.2f,%.2f) L=%.2f R=%.2f buttons=%s",
            mx, my, cx, cy, l_val, r_val, pressed
        )

# ════════════════════════════════════════════════════════════════════════════
# 6)  Dolphin loop
# ════════════════════════════════════════════════════════════════════════════
def signal_handler(sig, _):
    for c in controllers.values():
        c.disconnect()
    console.stop()
    log.info("Shutting down…")
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
        log.error("Console connect failed"); sys.exit(1)
    for c in controllers.values():
        if not c.connect():
            log.error("Controller connect failed"); sys.exit(1)
    log.info("Console + controllers connected.")

    rows: deque[Dict[str, Any]] = deque(maxlen=ROLL_WIN)
    start_wait=0
    while True:
        gs = console.step()
        if gs is None:
            continue

        # menu helper (unchanged)
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            c0, c1 = controllers[ports[0]], controllers[ports[1]]
            melee.MenuHelper().menu_helper_simple(
                gs, c0, melee.Character.FALCO, melee.Stage.FINAL_DESTINATION,
                cpu_level=0, autostart=0
            )
            melee.MenuHelper().menu_helper_simple(
                gs, c1, melee.Character.FALCO, melee.Stage.FINAL_DESTINATION,
                cpu_level=0, autostart=1
            )
            continue

        # ---------- build row ----------
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

        # ---------- projectiles ----------
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

        # ---------- inference ----------
        rows.append(row)
        if len(rows) == ROLL_WIN:
            start_wait+=1
            if start_wait>0:
                pred = run_inference(list(rows))
                ctrl = controllers[ports[0]]
                ctrl.release_all()
                press_output(ctrl, pred)
                ctrl.flush()

                # always log every frame
                mx, my = map(float, pred["main_xy"].tolist())
                dir_idx = int(torch.argmax(pred["c_dir_logits"]))
                cx, cy = C_DIR_TO_FLOAT.get(dir_idx, (0.5, 0.5))
                l_val = _safe(pred["L_val"].item(), 0.0)
                r_val = _safe(pred["R_val"].item(), 0.0)
                idx_to_button = [
                    melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
                    melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
                    melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
                    melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_START,
                    melee.enums.Button.BUTTON_D_UP, melee.enums.Button.BUTTON_D_DOWN,
                    melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,
                ]
                pressed = [btn.name for prob, btn in zip(pred["btn_probs"], idx_to_button) if prob.item() > 0.5]
                log.info(
                    "[%d] MAIN=(%.2f,%.2f) C=(%.2f,%.2f) L=%.2f R=%.2f BUTTONS=%s",
                    gs.frame, mx, my, cx, cy, l_val, r_val, pressed
                )
